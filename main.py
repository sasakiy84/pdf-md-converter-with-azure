"""
PDF ファイルを Azure の OCR で解析して、markdown に変換するスクリプト
markdown ファイルは、pandoc で epub に変換することを想定している.
画像や表、数式は、それぞれの画像を保存して、markdown にリンクを追加する

具体的な手順は以下の通り

0. 事前準備
```
pip install azure-ai-documentintelligence pymupdf
```

また、AZURE_TEXTBOOK_IMAGE_RESEARCH_API_KEY と AZURE_TEXTBOOK_IMAGE_RESEARCH_ENDPOINT を secret.py に保存しておく

1. PDF ファイルを Azure の OCR で解析する
```
python main.py azure /path/to/pdf_file.pdf
```

数式が含まれている場合は、--use-formula-addon オプションをつけることで、数式の解析を行うことができる。
数式がないときはデフォルトの方が見やすい。

results ディレクトリに、解析結果が保存される

2. 解析結果を markdown に変換する
```
python main.py markdown /path/to/azure_result_folder
```

3. pandoc で epub に変換する
```
cd /path/to/azure_result_folder
pandoc result.md -o result.epub --toc --epub-cover-image=figures/cover.png  --metadata title='title' --css=epub.css
```

heading の深さをうまくとれなかったりと、完全な変換は難しいので、
2 で生成された markdown を手で修正する必要がある
"""

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Optional, Union, Literal
from azure.core.pipeline.policies import RetryPolicy
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    DocumentTable,
    DocumentSection,
    DocumentParagraph,
    DocumentFigure,
    ParagraphRole,
    DocumentAnalysisFeature,
    DocumentContentFormat,
    DocumentFormula,
    DocumentFormulaKind,
)
import pymupdf
import argparse
from logging import Logger, getLogger, DEBUG

from secret import AZURE_TEXTBOOK_IMAGE_RESEARCH_API_KEY, AZURE_TEXTBOOK_IMAGE_RESEARCH_ENDPOINT

logger = getLogger(__name__)
logger.setLevel(DEBUG)

endpoint = AZURE_TEXTBOOK_IMAGE_RESEARCH_ENDPOINT
key = AZURE_TEXTBOOK_IMAGE_RESEARCH_API_KEY
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key),
    retry_policy=RetryPolicy(timeout=100, retry_total=3),
)


MODEL_ID = "prebuilt-layout"


def escape_html_tags(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def analyze_document_with_azure(
    pdf_file_path: Path, result_folder_path: Path, use_formula_addon: bool = False
):
    pdf_reader = open(pdf_file_path, "rb")
    addons: list[DocumentAnalysisFeature] = []
    if use_formula_addon:
        addons.append(DocumentAnalysisFeature.FORMULAS)
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout",
        pdf_reader,
        output_content_format=DocumentContentFormat.TEXT,
        features=addons,
    )
    pdf_reader.close()
    result: AnalyzeResult = poller.result()

    result_folder_path.mkdir(exist_ok=True, parents=True)
    with open(result_folder_path / "response.json", "w") as f:
        f.write(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


class DocGennerator:
    """
    Azure の OCR データを markdown に変換するクラス

    ```python
    result_content: AnalyzeResult = AnalyzeResult(json.loads(analyzed_json_path.read_text()))
    doc_generator = DocGennerator(result_content, pdf_file_path, analyzed_json_path.parent)
    markdown = doc_generator.gen()
    doc_generator.save_cover_image(cover_page_number)
    ```
    """

    def __init__(
        self,
        result: AnalyzeResult,
        pdf_file_path: Path,
        result_folder_path: Path,
        *,
        logger: Logger = getLogger(__name__),
        table_mode: Literal["image", "markdown", "image_with_comment_md"] = "image_with_comment_md",
    ):
        """
        result は、azure の OCR データでテキストや座標などの情報が含まれている
        pdf_file_path は、入力 pdf ファイルのパスで、画像や表紙を取得するために使用する
        result_folder_path は、出力するファイルのパスで、markdown や epub などのファイルを保存する
        table_mode は、表の出力形式を指定する。
        image は画像として表を保存する。
        markdown はマークダウン形式で表を保存する。
        image_with_comment_md は画像として表を保存し、その後にコメントとしてマークダウン形式のテーブルを埋め込む。
        """
        self.result = result
        self.content = result.content
        self.pages = result.pages
        self.sections = result.sections if result.sections is not None else []
        self.tables = result.tables if result.tables is not None else []
        self.paragraphs = result.paragraphs if result.paragraphs is not None else []
        self.figures = result.figures if result.figures is not None else []
        self.pdf_pages = pymupdf.Document(pdf_file_path)
        self.asset_folder_path = result_folder_path / "figures"
        self.asset_folder_path.mkdir(exist_ok=True, parents=True)
        self.logger = logger
        self.table_mode = table_mode

        self.formulas: list[DocumentFormula] = []
        for page in self.pages:
            if page.formulas is not None:
                self.formulas.extend(page.formulas)

        self.formula_inserted_paragraph_content: dict[int, str] = defaultdict(str)

        self._current_section_depth = 0
        self._table_number_counter = 0
        self._already_processed_section_ids: set[str] = set()

        # formula の数が一致しないことがあるので、チェックする
        # もし一致しなかったら、間違いを防ぐためにそのページの formula のリストをクリアして、そのページの formula は無視することにする
        formula_mark_count_per_page = defaultdict(int)
        for paragraph in self.paragraphs:
            count = paragraph.content.count(":formula:")
            page_number = paragraph.bounding_regions[0].page_number
            formula_mark_count_per_page[page_number - 1] += count
        self.formulas_per_page: dict[int, list[DocumentFormula]] = defaultdict(list)
        for page in self.pages:
            # 0-indexed にする
            page_number = page.page_number - 1
            if page.formulas is not None:
                self.formulas_per_page[page_number] += page.formulas

            if formula_mark_count_per_page[page_number] != len(
                self.formulas_per_page[page_number]
            ):
                self.logger.warning(
                    f"formula count mismatch: {formula_mark_count_per_page[page_number]=} != {len(self.formulas_per_page[page_number])=}, page {page_number}"
                )
                self.formulas_per_page[page_number] = []

    def _prepare_formula_inserted_text_dict(self) -> None:
        """
        formula が挿入された段落の text を保持する dict を作成する
        文中に現れる :formula: の対応先を、そのパラグラフだけで同定できないため、事前に変換した text を保持しておく
        """
        prev_page_number = 0
        formula_offset_in_page = 0
        for paragraph_id, paragraph in enumerate(self.paragraphs):
            if ":formula:" in paragraph.content:
                markdown = escape_html_tags(paragraph.content)
                current_page = paragraph.bounding_regions[0].page_number - 1
                if current_page != prev_page_number:
                    formula_offset_in_page = 0
                formulas_in_current_page = self.formulas_per_page[current_page]
                while ":formula:" in markdown:
                    # formula を見つけたら、順番に画像を保存して、markdown に置換する
                    if formula_offset_in_page >= len(formulas_in_current_page):
                        self.logger.warning(
                            f"formula offset exceeds the number of formulas in page {current_page}"
                        )
                        break

                    formula = formulas_in_current_page[formula_offset_in_page]
                    formula_file_name = (
                        f"formula_{current_page}_{formula_offset_in_page}.png"
                    )
                    formula_image = self._get_pixmap_from_bounding_region(
                        formula.polygon,
                        current_page + 1,
                        resolution_level=3.0,
                        margin_horizontal=2,
                        margin_vertical=1,
                    )
                    formula_image.save(self.asset_folder_path / formula_file_name)
                    if formula.kind == DocumentFormulaKind.INLINE:
                        markdown = re.sub(
                            r":formula:",
                            f"<div class='inline-formula-img'>![formula {current_page}_{formula_offset_in_page}](./figures/{formula_file_name}) \\\\\n</div>",
                            markdown,
                            count=1,
                        )
                    elif formula.kind == DocumentFormulaKind.DISPLAY:
                        markdown = re.sub(
                            r":formula:",
                            f"\n![formula {current_page}_{formula_offset_in_page}](./figures/{formula_file_name}) \\\\",
                            markdown,
                            count=1,
                        )
                    formula_offset_in_page += 1

                self.formula_inserted_paragraph_content[paragraph_id] = markdown
                prev_page_number = current_page

    def _get_element(
        self, element: str
    ) -> Optional[
        tuple[
            Union[DocumentTable, DocumentSection, DocumentParagraph, DocumentFigure],
            int,
        ]
    ]:
        """
        "/paragraphs/307" -> paragraphs[307]
        """
        _, element_type, _element_id = element.split("/")
        element_id = int(_element_id)
        if element_type == "sections":
            return self.sections[element_id], element_id
        elif element_type == "tables":
            return self.tables[element_id], element_id
        elif element_type == "paragraphs":
            return self.paragraphs[element_id], element_id
        elif element_type == "figures":
            return self.figures[element_id], element_id
        else:
            return None

    def _get_element_text(self, element: DocumentParagraph, paragraph_id: int) -> str:
        """
        element の text を取得する
        このとき、formula が挿入されている場合は、その formula の画像を取得して、markdown に追加する
        また、section heading の場合は、# を追加する
        """
        if not self._is_main_content_paragraph(element):
            return ""

        markdown_text = ""

        if (
            ":formula:" in element.content
            and paragraph_id in self.formula_inserted_paragraph_content
        ):
            markdown_text = self.formula_inserted_paragraph_content[paragraph_id]
        else:
            markdown_text = escape_html_tags(element.content)

        if (
            element.role == ParagraphRole.SECTION_HEADING
            or element.role == ParagraphRole.TITLE
        ):
            markdown_text = f"\n{'#' * self._current_section_depth} {markdown_text}\n"
        else:
            markdown_text = f"{markdown_text.replace('- ', '')}"

        return f"{markdown_text}\n\n"

    def _is_main_content_paragraph(self, paragraph: DocumentParagraph) -> bool:
        """
        本文の段落かどうかを判定する
        """
        role = paragraph.role
        roles_to_be_excluded = [
            ParagraphRole.PAGE_HEADER,
            ParagraphRole.PAGE_NUMBER,
            ParagraphRole.PAGE_FOOTER,
        ]
        return role not in roles_to_be_excluded

    def _get_pixmap_from_bounding_region(
        self,
        polygon: list[float],
        page_number: int,
        resolution_level: float = 3.0,
        margin_vertical: int = 5,
        margin_horizontal: int = 5,
    ) -> pymupdf.Pixmap:
        """
        bounding_region から pixmap を取得する
        resolution_level は、高いほど画像が大きくなる
        """
        normalized_bounding_box = self._calc_normalized_bounding_box(
            polygon, page_number
        )
        normalized_left, normalized_top, normalized_right, normalized_bottom = (
            self._get_left_top_right_bottom(normalized_bounding_box)
        )
        # azure の page number は 1-indexed なので、1 を引く
        current_page: pymupdf.Page = self.pdf_pages.load_page(page_number - 1)
        page_width, page_height = (
            current_page.bound().width,
            current_page.bound().height,
        )
        if page_width is None or page_height is None:
            raise ValueError(f"page {page_number} has no width or height")

        top = normalized_top * page_height - margin_vertical
        bottom = normalized_bottom * page_height + margin_vertical
        right = normalized_right * page_width + margin_horizontal
        left = normalized_left * page_width - margin_horizontal

        clip = pymupdf.Rect(left, top, right, bottom)
        return pymupdf.utils.get_pixmap(
            current_page,
            clip=clip,
            matrix=pymupdf.Matrix(resolution_level, resolution_level),
        )

    def _process_figure(self, figure: DocumentFigure) -> str:
        """
        figure の処理を行う.
        pymupdf を用いて、bounding_box から画像を切り出し、保存する。
        そして、そのリンクを markdown に追加する
        """
        markdown = ""

        # bounding box から画像を切り出し、保存する
        result_file_name = f"fig_{figure.id}.png"
        polygons: list[float] = []
        page_number: int | None = None
        for index, bounding_region in enumerate(figure.bounding_regions, 1):
            if page_number is None:
                page_number = bounding_region.page_number
            elif page_number != bounding_region.page_number:
                raise ValueError(
                    f"page number is different: {page_number} and {bounding_region.page_number}"
                )
            polygons.extend(bounding_region.polygon)

        if figure.caption is not None:
            for bounding_region in figure.caption.bounding_regions:
                if page_number != bounding_region.page_number:
                    raise ValueError(
                        f"page number is different: {page_number} and {bounding_region.page_number}"
                    )
                polygons.extend(bounding_region.polygon)

        image = self._get_pixmap_from_bounding_region(polygons, page_number)
        image.save(self.asset_folder_path / result_file_name)
        markdown += f"\n\n![{figure.id}](./figures/{result_file_name}) \\\n\n"

        # footnote があれば追加する
        footnotes = figure.footnotes if figure.footnotes is not None else []
        for i, footnote in enumerate(footnotes):
            for bounding_region in footnote.bounding_regions:
                page_number = bounding_region.page_number
                image = self._get_pixmap_from_bounding_region(
                    bounding_region.polygon, page_number
                )
                image.save(self.asset_folder_path / f"fig_{figure.id}_footnote_{i}.png")
                markdown += f"\n\n![{figure.id} footnote {i}](./figures/fig_{figure.id}_footnote_{i}.png) \\\n\n"

        return markdown

    def _process_table(self, table: DocumentTable) -> str:
        """
        table の処理を行う.
        table_mode に応じて、画像として保存するか、マークダウン形式で出力するかを選択する
        """
        if self.table_mode == "image":
            return self._process_table_as_image(table)
        elif self.table_mode == "markdown":
            return self._process_table_as_markdown(table)
        elif self.table_mode == "image_with_comment_md":
            return self._process_table_as_image_with_comment_md(table)
        else:
            raise ValueError(f"Invalid table_mode: {self.table_mode}")

    def _process_table_as_image(self, table: DocumentTable) -> str:
        """
        table を画像として保存する
        """
        markdown = ""
        page_number: int | None = None
        polygons: list[float] = []
        result_file_path = (
            self.asset_folder_path / f"table_{self._table_number_counter}.png"
        )
        for bounding_region in table.bounding_regions:
            if page_number is None:
                page_number = bounding_region.page_number
            elif page_number != bounding_region.page_number:
                raise ValueError(
                    f"page number is different: {page_number} and {bounding_region.page_number}"
                )
            polygons.extend(bounding_region.polygon)

        if table.caption is not None:
            for bounding_region in table.caption.bounding_regions:
                if page_number != bounding_region.page_number:
                    raise ValueError(
                        f"page number is different: {page_number} and {bounding_region.page_number}"
                    )
                polygons.extend(bounding_region.polygon)

        image = self._get_pixmap_from_bounding_region(polygons, page_number)
        image.save(result_file_path)
        markdown += f"\n\n![table {self._table_number_counter}](./figures/table_{self._table_number_counter}.png) \\\n\n"

        # footnote があれば追加する
        footnotes = table.footnotes if table.footnotes is not None else []
        for i, footnote in enumerate(footnotes):
            for bounding_region in footnote.bounding_regions:
                page_number = bounding_region.page_number
                image = self._get_pixmap_from_bounding_region(
                    bounding_region.polygon, page_number
                )
                image.save(
                    self.asset_folder_path
                    / f"table_{self._table_number_counter}_footnote_{i}.png"
                )
                markdown += f"\n\n![table {self._table_number_counter} footnote {i}](./figures/table_{self._table_number_counter}_footnote_{i}.png) \\\n\n"

        self._table_number_counter += 1
        return markdown

    def _process_table_as_markdown(self, table: DocumentTable) -> str:
        """
        table をマークダウン形式で出力する
        """
        markdown = "\n\n"
        
        # キャプションがあれば追加
        if table.caption is not None and table.caption.content:
            markdown += f"**{table.caption.content}**\n\n"

        # テーブルのヘッダーとセルを取得
        if not table.cells:
            return markdown

        # 列数を取得
        max_col = max(cell.column_index for cell in table.cells)
        max_row = max(cell.row_index for cell in table.cells)

        # テーブルのヘッダー行を取得
        header_cells = [cell for cell in table.cells if cell.kind == "columnHeader"]
        if not header_cells:
            # ヘッダー行がない場合は最初の行をヘッダーとして扱う
            header_cells = [cell for cell in table.cells if cell.row_index == 0]

        # ヘッダー行を作成
        header_row = [""] * (max_col + 1)
        for cell in header_cells:
            header_row[cell.column_index] = cell.content

        # ヘッダー行を出力
        markdown += "| " + " | ".join(header_row) + " |\n"
        markdown += "| " + " | ".join(["---"] * (max_col + 1)) + " |\n"

        # データ行を出力
        for row in range(1 if header_cells else 0, max_row + 1):
            row_cells = [cell for cell in table.cells if cell.row_index == row]
            if not row_cells:
                continue

            row_data = [""] * (max_col + 1)
            for cell in row_cells:
                row_data[cell.column_index] = cell.content

            markdown += "| " + " | ".join(row_data) + " |\n"

        # フットノートがあれば追加
        footnotes = table.footnotes if table.footnotes is not None else []
        for i, footnote in enumerate(footnotes):
            if footnote.content:
                markdown += f"\n^{i+1}: {footnote.content}\n"

        markdown += "\n"
        return markdown

    def _process_table_as_image_with_comment_md(self, table: DocumentTable) -> str:
        """
        table を画像として保存し、その後にコメントとしてマークダウン形式のテーブルを埋め込む
        """
        image_markdown = self._process_table_as_image(table)
        md_table = self._process_table_as_markdown(table)
        
        # マークダウンテーブルをHTMLコメントとして埋め込む
        comment_md = f"\n<!--\nMarkdown table version:\n{md_table}\n-->\n"
        
        return image_markdown + comment_md

    def _process_section(
        self,
        section: DocumentSection,
        section_id: int,
        section_depth_addition: bool = True,
    ) -> str:
        """
        section の処理を行う
        """
        if section_id in self._already_processed_section_ids:
            return ""

        if section_depth_addition:
            self._current_section_depth += 1
        markdown = ""
        for element_matcher in section.elements:
            print(f"processing element {element_matcher}")
            element, element_id = self._get_element(element_matcher)
            if isinstance(element, DocumentParagraph):
                markdown += self._process_paragraph(element, element_id)
            elif isinstance(element, DocumentFigure):
                markdown += self._process_figure(element)
            elif isinstance(element, DocumentTable):
                markdown += self._process_table(element)
            elif isinstance(element, DocumentSection):
                markdown += self._process_section(element, element_id)

        if section_depth_addition:
            self._current_section_depth -= 1
        self._already_processed_section_ids.add(section_id)
        return markdown

    def _process_paragraph(
        self, paragraph: DocumentParagraph, paragraph_id: int
    ) -> str:
        """
        paragraph の処理を行う
        """
        self.logger.info("processing paragraph")
        markdown = self._get_element_text(paragraph, paragraph_id)
        return markdown

    def _calc_normalized_bounding_box(
        self, bounding_box: list[float], page_number: int
    ) -> list[float]:
        """
        bounding box の座標を正規化する
        """
        page_width, page_height = (
            self.pages[page_number - 1].width,
            self.pages[page_number - 1].height,
        )
        if page_width is None or page_height is None:
            raise ValueError(f"page {page_number} has no width or height")

        normalized_bounding_box = []
        for i, coord in enumerate(bounding_box):
            if i % 2 == 0:
                normalized_bounding_box.append(coord / page_width)
            else:
                normalized_bounding_box.append(coord / page_height)
        return normalized_bounding_box

    def _get_left_top_right_bottom(
        self, bounding_box: list[float]
    ) -> tuple[float, float, float, float]:
        """
        bounding box の座標から、左上の x, y 座標と右下の x, y 座標を取得する
        すべての領域がカバーされるようにする
        """
        left_top: tuple[float, float] = (bounding_box[0], bounding_box[1])
        right_bottom: tuple[float, float] = (bounding_box[4], bounding_box[5])
        for x, y in zip(bounding_box[::2], bounding_box[1::2]):
            left_top = (min(left_top[0], x), min(left_top[1], y))
            right_bottom = (max(right_bottom[0], x), max(right_bottom[1], y))
        return left_top[0], left_top[1], right_bottom[0], right_bottom[1]

    def gen(self) -> str:
        """
        Azure の OCR データ を markdown に変換する
        セクションの深さに応じて、# の数を変える
        数式、図、表については、画像を保存して、リンクを markdown に追加する
        """
        self._already_processed_section_ids = set()
        self._prepare_formula_inserted_text_dict()
        markdown = ""
        for i, section in enumerate(self.sections, 1):
            if i in self._already_processed_section_ids:
                continue
            self.logger.info(f"processing section {i}/{len(self.sections)}")
            # 最初の section は section_depth_addition を行わないことで、いい感じの見た目になる
            markdown += self._process_section(
                section, section_id=i, section_depth_addition=False
            )
        return markdown

    def save_cover_image(self, cover_page_number: int) -> None:
        """
        cover image を保存する
        """
        print(f"cover page number: {cover_page_number}")
        cover_image = pymupdf.utils.get_page_pixmap(self.pdf_pages, cover_page_number)
        cover_image.save(self.asset_folder_path / "cover.png")


def construct_markdown_from_result(
    analyzed_json_path: Path,
    pdf_file_path: Path,
    cover_page_number: int | None = 0,
    table_mode: Literal["image", "markdown", "image_with_comment_md"] = "image_with_comment_md",
) -> str:
    """
    section を上から順に見ていき、順番に markdown に変換していく
    そのとき、
    """
    result_content: AnalyzeResult = AnalyzeResult(
        json.loads(analyzed_json_path.read_text())
    )
    doc_generator = DocGennerator(
        result_content,
        pdf_file_path,
        analyzed_json_path.parent,
        table_mode=table_mode,
    )
    if cover_page_number is not None:
        doc_generator.save_cover_image(cover_page_number)
    return doc_generator.gen()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers(help="subcommand help")

    def ocr_command(args: argparse.Namespace):
        pdf_file_path: Path = args.pdf_file_path
        result_root_folder_path: Path = args.resultdir
        result_folder_path = (
            result_root_folder_path / pdf_file_path.with_suffix("").name
        )
        result_folder_path.mkdir(exist_ok=True, parents=True)
        (result_folder_path / "source.pdf").write_bytes(pdf_file_path.read_bytes())
        logger.info(f"analyzing {pdf_file_path}")

        analyze_document_with_azure(
            pdf_file_path, result_folder_path, args.use_formula_addon
        )
        logger.info(f"result is saved in {result_folder_path}")

    azure_ocr_parser = subparsers.add_parser("azure")
    azure_ocr_parser.add_argument("pdf_file_path", type=Path)
    azure_ocr_parser.add_argument("--resultdir", type=Path, default=Path("results"))
    azure_ocr_parser.add_argument("--use-formula-addon", type=bool, default=False)
    azure_ocr_parser.set_defaults(func=ocr_command)

    def markdown_command(args: argparse.Namespace):
        result_folder_path: Path = args.result_folder_path
        json_file_path: Path = result_folder_path / "response.json"
        pdf_file_path: Path = (
            args.pdf_file_path
            if args.pdf_file_path
            else result_folder_path / "source.pdf"
        )
        cover_page = args.cover_page if not args.no_cover else None
        markdown = construct_markdown_from_result(
            json_file_path,
            pdf_file_path,
            cover_page,
            table_mode=args.table_mode,
        )
        result_markdown_file_path = result_folder_path / "result.md"
        result_markdown_file_path.write_text(markdown)

        epub_css = """
.inline-formula-img {
    display: inline-block;
    height: 1.2em;
    line-height: 1.2;
    margin: 0;
    padding: 0;
}
"""
        epub_css_file_path = result_folder_path / "epub.css"
        epub_css_file_path.write_text(epub_css)

        print(
            f"cd {result_folder_path} && pandoc result.md -o {result_folder_path.with_suffix('.epub').name} --toc --epub-cover-image=figures/cover.png  --metadata title='{result_folder_path.name}' --css=epub.css --strip-comments"
        )

    markdown_parser = subparsers.add_parser("markdown")
    markdown_parser.add_argument("result_folder_path", type=Path)
    markdown_parser.add_argument("--pdf_file_path", type=Path, default=None)
    markdown_parser.add_argument("--cover_page", type=int, default=0)
    markdown_parser.add_argument("--no-cover", type=bool, default=False)
    markdown_parser.add_argument(
        "--table-mode",
        type=str,
        choices=["image", "markdown", "image_with_comment_md"],
        default="image_with_comment_md",
        help="Table output mode: 'image' for image-based tables, 'markdown' for markdown tables, 'image_with_comment_md' for both formats (default)",
    )
    markdown_parser.set_defaults(func=markdown_command)

    args = argparser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        argparser.print_help()
