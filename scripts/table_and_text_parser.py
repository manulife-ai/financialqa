#!/usr/bin/python3
import argparse
import os
from typing import Dict, List

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import numpy as np
import pandas as pd

from dotenv import load_dotenv

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

load_dotenv() # load environment variables from .env (add to .gitignore) 

def extract_blob_paths(container_client):

    list_of_blob_paths = []
    
    for i, blob in enumerate(container_client.list_blobs()):
        path = "https://" + os.environ["AZURE_STORAGE_CONTAINER_ACCOUNT"] + \
            ".blob.core.windows.net/" + os.environ['AZURE_STORAGE_CONTAINER_NAME'] + "/" + blob.name
        list_of_blob_paths.append(path)
    
    return list_of_blob_paths

def parse_pdfs(list_of_blob_paths: str) -> List[Dict[str, str]]:

    result_dicts = []

    for blob_path in list_of_blob_paths:

        endpoint = os.environ["DOCUMENT_ENDPOINT"]
        key = os.environ["DOCUMENT_KEY"]
        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
            )

        if blob_path.startswith("http"):
            poller = document_analysis_client.begin_analyze_document_from_url(
                "prebuilt-layout", blob_path)
        else:
            with open(blob_path, "rb") as f:
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-layout", document=f)

        result = poller.result()
        # Returns a dict representation of AnalyzeResult.
        result_dict = result.to_dict()
        result_dict['report_name'] = blob_path.split("/")[-1]
        result_dicts.append(result_dict)

    return result_dicts


def page_text_and_tables(result_dicts):

    page_content = {}
    page_contents = []

    # for dict in result_dict:
    for result_dict in result_dicts:

        for i, paragraph in enumerate(result_dict.get('paragraphs')):

            # print(paragraph.get('bounding_regions')[0].get('page_number'))
            # print(page_content.keys())
            if paragraph.get('bounding_regions')[0].get('page_number') in page_content.keys():
                # print(paragraph.get('bounding_regions')[0].get('page_number'))
                # print(i, page_content[i])
                page_content[paragraph.get('bounding_regions')[0].get('page_number')].\
                                get('text').append(paragraph.get('content'))
                # pass
            else:
                # print(i)
                page_content[paragraph.get('bounding_regions')[0].get('page_number')] = \
                    {'tables': [], 'text': [paragraph.get('content')]}
                # print(paragraph.get('bounding_regions')[0].get('page_number'))
                # print(page_content.keys())

        for idx, atable in enumerate(result_dict["tables"]):

            row_count = atable["row_count"]
            column_count = atable["column_count"]
            arr = np.empty((row_count, column_count), dtype=object)
            arr[0][:] = ""
            for aval in atable["cells"]:
            # Handles complex headers
                if aval["kind"] == "columnHeader":
                    arr[0][aval["column_index"]:aval["column_index"] +
                        aval["column_span"]] += str(aval["content"])
                else:
                    # Add edge cases here (later modularize)
                    arr[aval["row_index"]][aval["column_index"]] = aval["content"]
                    # print(arr[aval["row_index"]])
                    # if np.all(arr[aval["row_index"][0:]]) == None or np.all(arr[aval["row_index"][1:]]) == '':
                    #   print(arr[aval["row_index"]])
                    # return

            df = pd.DataFrame(arr)
            # print(df)
            df.columns = df.iloc[0]
            df = df.drop(df.index[0:2])
            df.reset_index(inplace=True, drop=True)
            df.dropna(inplace=True)
            # print(df)
            # if idx == 1:
            #     return
            page_content[atable.get('bounding_regions')[0].get('page_number')].get('tables').append(df)
            # tables.append(df)

        page_contents.append(page_content)

    return page_contents

def dedupe_text(page_content):

    for i, report in enumerate(paged_text_and_tables):
        num_pages = max(report.keys())
        if i != 2:
            continue
        for page_num, tables_and_text in report.items():
            for table in tables_and_text.get('tables'):
                print(type(tables_and_text.get('text')))
                print([tables_and_text.get('text') for text in tables_and_text.get('text') if text not in table.values])
                print(tables_and_text.get('text') for text in tables_and_text.get('text') if text not in table.values)

if __name__ == '__main__':
    pass