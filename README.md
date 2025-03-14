# PDF から EPUB へのコンバーター

このプロジェクトは、Azure の OCR サービスの結果からマークダウンを生成します。
主に、pandoc で EPUB に変換することを想定していますが、すこしいじれば他の用途にも使えるはずです。

## 前提条件

必要なパッケージをインストールします:
```sh
pip install azure-ai-documentintelligence pymupdf
```

以下の変数を `secret.py` に設定します:
```python
AZURE_TEXTBOOK_IMAGE_RESEARCH_API_KEY = ""
AZURE_TEXTBOOK_IMAGE_RESEARCH_ENDPOINT = ""
```

## 使用方法

### 1. Azure OCR で PDF を解析
```sh
python main.py azure /path/to/pdf_file.pdf
```
数式を解析するには `--use-formula-addon` オプションを使用します。

### 2. OCR 結果を Markdown に変換
```sh
python main.py markdown /path/to/azure_result_folder
```

### 3. Pandoc を使用して Markdown を EPUB に変換
```sh
cd /path/to/azure_result_folder
pandoc result.md -o result.epub --toc --epub-cover-image=figures/cover.png --metadata title='title' --css=epub.css
```

## 注意事項

- 生成された Markdown は手動で調整が必要な場合があります。
- スクリプトは OCR 結果を `results` ディレクトリに保存します。
