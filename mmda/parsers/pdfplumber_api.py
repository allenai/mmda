import os
import tempfile
from flask import Flask, request
from gevent.pywsgi import WSGIServer

from mmda.parsers.pdfplumber_parser import PDFPlumberParser
from mmda.rasterizers.rasterizer import PDF2ImageRasterizer


app = Flask(__name__)
pdfplumber = PDFPlumberParser()
rasterizer = PDF2ImageRasterizer()


@app.post("/")
def parse_pdf():
    with tempfile.TemporaryDirectory() as tempdir:
        pdf_path = f"{tempdir}/input.pdf"
        with open(pdf_path, "wb") as f:
            f.write(request.get_data())
        doc = pdfplumber.parse(pdf_path)
        doc.annotate_images(rasterizer.rasterize(input_pdf_path=pdf_path, dpi=72))
        return doc.to_json(with_images=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    http_server = WSGIServer(("", port), app)
    http_server.serve_forever()
