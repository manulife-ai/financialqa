import os
import sys
import logging

import numpy as np
import pandas as pd

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentAnalysisFeature

"""
To-do's:
    - Fix improper nested column headers
"""

logging.basicConfig(
    # filename=logfile,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s line %(lineno)d  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

azure_docintel_key = os.getenv('AZURE_DOCINTEL_KEY')
azure_docintel_endpoint = os.getenv('AZURE_DOCINTEL_ENDPOINT')
azure_docintel_version = os.getenv('AZURE_DOCINTEL_VERSION')


def extract_pdf_contents(pdfs_folder: str) -> dict[str, dict]:
    """
    Extract PDF contents using Azure Document Intelligence.

    Args:
        pdf_paths (list): List of PDF files to extract content from.

    Returns:
        dict: Dictionary containing extracted PDF contents. 
    """
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=azure_docintel_endpoint,
        credential=AzureKeyCredential(azure_docintel_key),
        version=azure_docintel_version,
    )
    pdf_contents_dict = {}
    pdf_paths = [os.path.join(pdfs_folder, pdf) for pdf in os.listdir(pdfs_folder)]
    for pdf_path in pdf_paths:
        if not pdf_path.endswith('.pdf'):
            continue
        pdf_name = pdf_path.split('/')[-1].replace('.pdf', '')
        pdf_name_parts = pdf_name.split('_')
        company_name = pdf_name_parts[0]
        report_quarter = pdf_name_parts[-1]
        logger.info(f"Extracting contents from '{pdf_name}'...")
        with open(pdf_path, "rb") as f:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout",
                body=f,
                features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS],
                content_type="application/octet-stream",
            )
            pdf_analysis_results: AnalyzeResult = poller.result()
        pdf_contents_dict[pdf_name] = {
            'company_name': company_name,
            'report_quarter': report_quarter,
            'docintel_analysis_results': pdf_analysis_results,
        }
    return pdf_contents_dict


def page_pdf_contents(pdf_contents_dict: dict[str, dict]) -> dict:
    """
    Create a dictionary of extracted PDF contents per page of each PDF.
    
    Args:
        pdf_contents_dict (dict): Dictionary containing extracted PDF 
            contents.

    Returns:
        dict: Dictionary containing extracted PDF contents per page of 
            each PDF.
    """
    pdf_pages_dict = dict()
    for pdf_name, pdf_items in pdf_contents_dict.items():
        pdf_analysis_results = pdf_items.get('docintel_analysis_results')
        company_name = pdf_items.get('company_name')
        report_quarter = pdf_items.get('report_quarter')
        pdf_contents = {
            'pages': {},
            'company_name': company_name,
            'report_quarter': report_quarter,
        }
        for paragraph in pdf_analysis_results.get('paragraphs'):
            page_num = paragraph.get('boundingRegions')[0].get('pageNumber')
            if page_num in pdf_contents['pages'].keys():
                pdf_contents['pages'][page_num].get('text').append(
                    paragraph.get('content'))
            else:
                pdf_contents['pages'][page_num] = {
                    'text': [paragraph.get('content')], 
                    'tables': []
                }
            if 'role' in paragraph.keys():
                if paragraph['role'] in pdf_contents['pages'][page_num].keys():
                    pdf_contents['pages'][page_num][paragraph['role']].append(
                        paragraph.get('content'))
                else:
                    pdf_contents['pages'][page_num][paragraph['role']] = [
                        paragraph.get('content')]
                # Remove duplicate text roles from text
                for role in pdf_contents['pages'][page_num][paragraph['role']]:
                    if role in pdf_contents['pages'][page_num]['text']:
                        pdf_contents['pages'][page_num]['text'].remove(role)

        for table in pdf_analysis_results.get('tables'):
            page_num = table.get('boundingRegions')[0].get('pageNumber')
            row_count = table['rowCount']
            column_count = table['columnCount']
            arr = np.empty((row_count, column_count), dtype=object)
            arr[0][:] = ''
            for cell in table['cells']:
                # Remove duplicate table cell values from text
                if cell['content'] in pdf_contents['pages'][page_num]['text']:
                    pdf_contents['pages'][page_num]['text'].remove(cell['content'])
                if 'kind' in cell.keys():
                    # Handles nested headers
                    if cell['kind'] == 'columnHeader':
                        if cell.get('spans'):
                            column_span_length = cell.get('spans')[0].get('length')
                        else:
                            column_span_length = 0
                        arr[0][
                            cell['columnIndex']:
                            cell['columnIndex'] + column_span_length
                        ] += ' ' + str(cell['content'])
                else:
                    arr[cell['rowIndex']][cell['columnIndex']] = cell['content']
            df = pd.DataFrame(arr)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df.reset_index(inplace=True, drop=True)
            df.dropna(inplace=True)
            df = df.to_dict(orient="records")
            if page_num in pdf_contents['pages'].keys():
                pdf_contents['pages'][page_num].get('tables').append(df)
            else:
                pdf_contents['pages'][page_num] = {'text': [], 'tables': [df]}
        pdf_pages_dict.update({pdf_name: pdf_contents})

        for figures in pdf_analysis_results.get('figures'):
            figure_bounding_regions = figures.get('boundingRegions')
            for i, bounding_regions in enumerate(
                    figure_bounding_regions, start=1
                    ):
                bounding_regions_polygon = bounding_regions.get('polygon')
                page_num = bounding_regions.get('pageNumber')
                if page_num not in pdf_contents.get('pages').keys():
                    pdf_contents['pages'].update({page_num: {'figures': {}}})
                elif 'figures' not in pdf_contents.get('pages').get(page_num):
                    pdf_contents['pages'].get(page_num).update({'figures': {}})
                figure_file_name = f'figure_{i}_page_{page_num}'
                if 'caption' in figures.keys():
                    pdf_contents['pages'][page_num].get('figures').update({
                        figure_file_name: {
                            'bounding_regions_polygon': bounding_regions_polygon,
                            'caption': figures.get('caption').get('content'),
                        }})
                else:
                    pdf_contents['pages'][page_num]['figures'].update({
                        figure_file_name: {
                            'bounding_regions_polygon': bounding_regions_polygon,
                        }})
    return pdf_pages_dict