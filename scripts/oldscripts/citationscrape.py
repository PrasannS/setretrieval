### File to scrape set retrieval dataset for citations
# for 10 different fields of study
#   get 10 different papers from each field
#   for each paper get all the references
#   choose 5 papers to repeat process with
#   do this for 3 hops
#   get all the papers that are a part of this graph (full text)
# This should get us a reasonably sized dataset of papers (maybe 100k) 
# that we can then use as a datastore and to test some different things
import json
import os
import re
import requests
import wget
from tqdm import tqdm
import time

fields = [
    "Computer Science",
    "Biology",
    "Medicine",
    "Physics",
    "History",
    "Art",
    "Linguistics",
    "Philosophy",
    "Business",
    "Psychology"
]

# keep on going until there are at least minpapers with mincites
def calibrate_field(field, mincites=400, minpapers=100):
    endyear = ":2025"
    startyear = 2023
    while startyear > 2000:
        result = recent_field_papers(field, mincites=mincites, year=f"{startyear}{endyear}")
        try:
            if len(result['data']) >= minpapers:
                break
        except KeyError:
            print(result)
        startyear -= 1
    result['startyear'] = startyear
    return result

def recent_field_papers(field, mincites=400, year="2023:2025"):
    # get a list of recent papers from field with at least mincites citations
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    params = {
        'query': "find",
        'fieldsOfStudy': field,
        'minCitationCount': mincites, 
        'publicationDateOrYear': year,
        'limit': 50,
        'fields': "title,authors,year,externalIds,citationCount,referenceCount,fieldsOfStudy"
    }
    headers = {
        'Content-Type': 'application/json'
    }
    time.sleep(1)
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    return data

if __name__=="__main__":
    if False:
        fielddata = []
        for field in fields:
            fieldresult = calibrate_field(field, mincites=400, minpapers=100)
            print(f"Calibrating field {field}")
            fielddata.append(fieldresult)
        with open("starterpapers.json", "w") as f:
            json.dump(fielddata, f, indent=4)
    fielddata = json.load(open("starterpapers.json"))
    breakpoint()
