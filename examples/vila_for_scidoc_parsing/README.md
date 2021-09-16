# Use VILA for Scientific Documents Parsing

## Installation 

1. Install the necessary python dependencies
    ```bash
    pip install mmda # alternatively, you can pip install ../ -e
    pip install -r requirements.txt
    ```
2. Install poppler for rendering PDF images
3. Download the VILA models

    For now, please contact @shannons for downloading the weights. 

## Usage

```bash
python main.py --pdf-path <path-to-pdf-files> \
               --vila-type <hvila-or-vila> \
               --vila-model-path <path-to-vila-models> \
               --export-folder <the-folder-for-saving-visualizations>
```
It will run PubLayNet-based layout detection models to identify the layout structure, and run
VILA models based on the detected layouts to perform token classification. All detected layouts
and token categories will visualized and stored in the export-folder.