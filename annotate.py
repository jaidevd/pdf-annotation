#!/usr/bin/env python
# coding: utf-8
import fitz
import pandas as pd
import os
from io import BytesIO

op = os.path
ANNDIR = op.join(op.dirname(__file__), "annotations")
if not op.isdir(ANNDIR):
    os.mkdir(ANNDIR)


def page_ocr(filename, page):
    with fitz.open(filename) as doc:
        pg = doc[page - 1]
        ocr = pg.get_textpage_ocr().extractBLOCKS()
    df = pd.DataFrame(ocr,
                      columns=["x0", "y0", "x1", "y1", "text", "id", "type"])
    df['page'] = page
    df.drop(["id"], axis=1, inplace=True)
    df = df[df['type'] == 0]
    df['text'] = df['text'].str.replace('\x01', ' ').str.strip()
    df.index.name = 'box_id'
    return df.reset_index()


def ocr(handler):
    filename, page = handler.path_args
    page = int(page)

    # Check if annotations exist
    ann_path = op.join(ANNDIR, f'{filename}_ann.json')
    if not op.isfile(ann_path):
        df = page_ocr(filename, page)
        df.to_json(ann_path, orient='records', indent=2)
    else:
        df = pd.read_json(ann_path)
        # Check if page exists
        page_df = df[df['page'] == page]
        if len(page_df) == 0:
            page_df = page_ocr(filename, page)
            df = pd.concat([df, page_df], axis=0, ignore_index=True)
            df.to_json(ann_path, orient='records', indent=2)

    return df[df['page'] == page].to_json(orient="records")


def save(handler):
    filename, page = handler.path_args
    page = int(page)
    outfile = op.join(ANNDIR, f'{filename}_ann.json')
    df = pd.read_json(outfile)
    df.set_index(['page', 'box_id'], verify_integrity=True, inplace=True)
    xdf = pd.read_json(handler.request.body.decode())
    xdf['page'] = page
    xdf.set_index(['page', 'box_id'], verify_integrity=True, inplace=True)
    df.loc[xdf.index] = xdf
    df.reset_index().to_json(outfile, orient='records', indent=2)


def _export_pdf(filename, df):
    with fitz.open(filename) as doc:
        for page, boxes in df.groupby('page'):
            pg = doc[page - 1]
            for _, text in boxes.iterrows():
                pg.add_redact_annot(
                    fitz.Rect(text.x0, text.y0, text.x1, text.y1),
                    text.text, cross_out=False)
            pg.apply_redactions()
        buff = BytesIO()
        doc.save(buff)
    buff.seek(0)
    return buff.read()


def export(handler):
    etype = handler.path_args[0]
    filename = handler.get_argument('file')
    filepath = op.join(ANNDIR, f'{filename}_ann.json')
    df = pd.read_json(filepath)
    if etype == 'csv':
        handler.set_header('Content-Type', 'text/csv')
        handler.set_header('Content-Disposition',
                           f'attachment; filename={filename}_ann.csv')
        return df.to_csv(index=False)
    handler.set_header('Content-Type', 'application/pdf')
    handler.set_header('Content-Disposition',
                       f'attachment; filename={filename}_ann.pdf')
    return _export_pdf(filename, df)
