import os
import sys
sys.path.append('..')

import argparse
from typing import Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import numpy as np
import pandas as pd


def parse_pdfs(list_of_blob_paths: str) -> List[Dict[str, str]]:

    result_dicts = []

    for blob_path in list_of_blob_paths:

        endpoint = os.environ['DOCUMENT_ENDPOINT']
        key = os.environ['DOCUMENT_KEY']
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
            )

        if blob_path.startswith('http'):
            poller = document_analysis_client.begin_analyze_document_from_url(
                'prebuilt-layout', blob_path)
        else:
            with open(blob_path, 'rb') as f:
                poller = document_analysis_client.begin_analyze_document(
                    'prebuilt-layout', document=f)

        result = poller.result()
        result_dict = result.to_dict() # Returns a dict representation of AnalyzeResult.
        result_dicts.append(result_dict)

    return result_dicts


def page_text_and_tables(result_dicts):

    page_contents = []

    for result_dict in result_dicts:
        page_content = {}

        for paragraph in result_dict.get('paragraphs'):
            page_num = paragraph.get('bounding_regions')[0].get('page_number')

            if page_num in page_content.keys():
                page_content[page_num].get('text').append(paragraph.get('content'))
            else:
                page_content[page_num] = {'tables': [], 'text': [paragraph.get('content')]}

        for table in result_dict['tables']:
            page_num = table.get('bounding_regions')[0].get('page_number')
            row_count = table['row_count']
            column_count = table['column_count']
            arr = np.empty((row_count, column_count), dtype=object)
            arr[0][:] = ''

            for cell in table['cells']:
            # Handles nested headers
                if cell['kind'] == 'columnHeader':
                    arr[0][cell['column_index']:cell['column_index'] +
                        cell['column_span']] += ' ' + str(cell['content'])
                else:
                    arr[cell['row_index']][cell['column_index']] = cell['content']

            df = pd.DataFrame(arr)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0:2])
            df.reset_index(inplace=True, drop=True)
            df.dropna(inplace=True)

            if page_num not in page_content.keys():
                page_content[page_num] = {'tables': [df], 'text': []}
            else:
                page_content[page_num].get('tables').append(df)

            dedupe_text_from_tables(page_num, page_content, df)

        page_contents.append(page_content)
    
    return page_contents


def dedupe_text_from_tables(page_num, page_content, df):
    
    page_content[page_num].get('text')[:] = \
        [text for text in page_content[page_num].get('text') if text not in df.values]