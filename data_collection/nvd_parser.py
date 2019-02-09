import re
import requests
import zipfile
import io
import json


def download_nvd_data(nvd_year_list_to_download, full=True):
    nvd_json_version_dict = dict()
    r = requests.get('https://nvd.nist.gov/vuln/data-feeds#JSON_FEED')

    for filename in re.findall("nvdcve-1.0-[0-9]*\.json\.zip", r.text):
        if not full:
            if extract_year(filename) not in nvd_year_list_to_download:
                continue
        print("Downloading {}".format(filename))
        r_zip_file = requests.get("https://static.nvd.nist.gov/feeds/json/cve/1.0/" + filename, stream=True)
        zip_file_bytes = io.BytesIO()

        for chunk in r_zip_file:
            zip_file_bytes.write(chunk)

        zip_file = zipfile.ZipFile(zip_file_bytes)

        for json_filename in zip_file.namelist():
            print("Extracting {}".format(json_filename))
            json_raw = zip_file.read(json_filename).decode('utf-8')
            json_data = json.loads(json_raw)

            for entry in json_data['CVE_Items']:
                cveid = entry['cve']['CVE_data_meta']['ID']
                nvd_json_version_dict[cveid] = {'ref':[], 'cve':'', 'nvd':{}}

                for ref in entry['cve']['references']['reference_data']:
                    nvd_json_version_dict[cveid]['ref'].append(ref['url'])

                # for description in entry['cve']['description']:
                nvd_json_version_dict[cveid]['cve'] = entry['cve']['description']['description_data'][0]['value']

                for vendor_idx in list(range(len(entry['cve']['affects']['vendor']['vendor_data']))):
                    try:
                        vendor_name = entry['cve']['affects']['vendor']['vendor_data'][vendor_idx]['vendor_name']
                    except IndexError:
                        vendor_name = ""

                    try:
                        for pd in entry['cve']['affects']['vendor']['vendor_data'][vendor_idx]['product']['product_data']:
                            vv = []
                            pd_name = pd['product_name'].replace('_', ' ')
                            for vd in pd['version']['version_data']:
                                vv.append([vd['version_affected'].lower(), vd['version_value'].lower()])
                            if pd_name != '' and vv != []:
                                nvd_json_version_dict[cveid]['nvd'][(vendor_name + ' ' + pd_name).lower()] = transform_version(vv)
                    except IndexError:
                        pass
    return nvd_json_version_dict


def transform_version(version_list):
    new_version_list = []
    for range_point in version_list:
        version_range, point = range_point
        if version_range == '<=':
            new_version_list.append(point + ' and earlier')
        elif version_range == '=':
            new_version_list.append(point)
        else:
            print('ERROR! range is: ', version_range)
    return new_version_list


def extract_year(filename):
    return filename.split('-')[2].split('.')[0]
