import os
import sys
import logging
logging.basicConfig(    
        # filename=logfile,    
        level=logging.INFO,    
        format="%(asctime)s %(levelname)s %(name)s line %(lineno)d  %(message)s",    
        datefmt="%H:%M:%S")   
import argparse
from typing import Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def parse_pdfs(report_contents) -> List[Dict[str, str]]:
    """Parse PDF files from Azure Blob Storage container using Azure Document Intellignece."""
    result_dicts = []
    logger.info('Parsing PDFs...')
    logging.disable(logging.WARNING)

    for report_name, report_content in report_contents.items():
        endpoint = os.environ['DOCUMENT_ENDPOINT']
        key = os.environ['DOCUMENT_KEY']
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
            )
        report_blob_path = report_content.get('report_blob_path')

        if report_blob_path.startswith('http'):
            poller = document_analysis_client.begin_analyze_document_from_url(
                'prebuilt-layout', report_blob_path)
        else:
            with open(report_blob_path, 'rb') as f:
                poller = document_analysis_client.begin_analyze_document(
                    'prebuilt-layout', document=f)
        result = poller.result()
        result_dict = result.to_dict() # Returns a dict representation of AnalyzeResult.
        result_dict['report_name'] = report_name
        result_dict['company_name'] = report_content.get('company_name')
        result_dict['report_quarter'] = report_content.get('report_quarter')
        result_dict['report_blob_path'] = report_content.get('report_blob_path')
        result_dicts.append(result_dict)
    logging.disable(logging.NOTSET)
    return result_dicts

def page_text_and_tables(result_dicts):
    """Store extracted text and tables as values into a nested dictionary based on page number keys."""
    page_contents = []

    for result_dict in result_dicts:
        page_content = {'pages': {}}

        for paragraph in result_dict.get('paragraphs'):
            page_num = paragraph.get('bounding_regions')[0].get('page_number')
            if page_num in page_content['pages'].keys():
                page_content['pages'][page_num].get('text').append(paragraph.get('content'))
            else:
                page_content['pages'][page_num] = {'tables': [], 'text': [paragraph.get('content')]}

            if paragraph['role'] is not None:
                if paragraph['role'] in page_content['pages'][page_num].keys():
                    page_content['pages'][page_num][paragraph['role']].append(paragraph.get('content'))
                else:
                    page_content['pages'][page_num][paragraph['role']] = [paragraph.get('content')]
                # remove duplicates from text
                for role in page_content['pages'][page_num][paragraph['role']]:
                    if role in page_content['pages'][page_num]['text']:
                        page_content['pages'][page_num]['text'].remove(role)

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

            if page_num in page_content['pages'].keys():
                page_content['pages'][page_num].get('tables').append(df)
            else:
                page_content['pages'][page_num] = {'tables': [df], 'text': []}
            _dedupe_text_from_tables(page_num, page_content, df)

        page_content['report_name'] = result_dict['report_name']
        page_content['company_name'] = result_dict['company_name']
        page_content['report_quarter'] = result_dict['report_quarter']
        page_content['report_blob_path'] = result_dict['report_blob_path']
        page_contents.append(page_content)
    
    return page_contents


def _dedupe_text_from_tables(page_num, page_content, df):
    """Remove duplicate text from a table."""
    page_content['pages'][page_num].get('text')[:] = \
        [text for text in page_content['pages'][page_num].get('text') if text not in df.values]