import os
import tempfile
from flask import Flask, request
from gevent.pywsgi import WSGIServer

from mmda.parsers.symbol_scraper_parser import SymbolScraperParser

app = Flask(__name__)

sscraper = SymbolScraperParser("sscraper")

@app.post("/")
def parse_pdf():
    with tempfile.TemporaryDirectory() as tempdir:
        pdf_path = f"{tempdir}/input.pdf"
        with open(pdf_path, "wb") as f:
            f.write(request.get_data())
        doc = sscraper.parse(pdf_path)
        # TODO: add image stuff here
        return doc.to_json()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    http_server = WSGIServer(("", port), app)
    http_server.serve_forever()
