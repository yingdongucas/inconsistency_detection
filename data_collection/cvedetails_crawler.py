import requests
from bs4 import BeautifulSoup
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import utils
import multiprocessing
from report_crawler import crawl_report


def crawl_ref_for_one_vul_category(args):
    cat_file, cve_id_dir, cve_ref_dir = args
    cve_ref_dict = dict()
    f = open(cve_id_dir + '/' + cat_file, 'r')
    lines = f.readlines()
    for line in lines:
        idx_cve = line.split('\t')
        cve_id = idx_cve[1]
        link = 'https://cve.mitre.org/cgi-bin/cvename.cgi?name=' + cve_id
        ref_link_list = []
        try:
            page = requests.get(link, timeout=60, headers={'User-Agent': "Magic Browser"})
            content = BeautifulSoup(page.content).get_text()
            split_lines = content.split('\n')
            for line_l in split_lines:
                loc = line_l.find(':http')
                if loc == -1:
                    continue
                url = line_l[loc + 1:].strip()
                ref_link_list.append(url)
                # print(url)
            cve_ref_dict[cve_id] = ref_link_list

        except requests.exceptions.HTTPError as errh:
            print("Http Error: " + str(errh) + " Please check: " + link)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:" + str(errc) + " Please check: " + link)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:" + str(errt) + " Please check: " + link)
        except requests.exceptions.RequestException as err:
            print("Other errors!" + str(err) + " Please check: " + link)

        with open(cve_ref_dir + cat_file + '.py', 'w') as f_w:
            f_w.write('cve_ref_dict = ' + str(cve_ref_dict))


class Crawler:

    def __init__(self, cve_id_dir, cve_ref_dir, data_dir, num_of_CPUs):
        self.cve_id_dir = cve_id_dir
        self.cve_ref_dir = cve_ref_dir
        self.data_dir = data_dir
        self.pool = multiprocessing.Pool(num_of_CPUs)

    def run(self):
        self.crawl_cveid_list()
        self.crawl_cve_refs()
        self.crawl_reports_by_refs()

    def crawl_cveid_list(self):
        link = 'https://www.cvedetails.com/vulnerability-list.php?vendor_id=0&product_id=0&version_id=0&page=1&hasexp=0&opdos=1&opec=0&opov=0&opcsrf=0&opgpriv=0&opsqli=0&opxss=0&opdirt=0&opmemc=0&ophttprs=0&opbyp=0&opfileinc=0&opginf=0&cvssscoremin=0&cvssscoremax=0&year=0&month=0&cweid=0&order=1&trc=21357&sha=38745b427397c23f6ed92e0ed2d3e114da828672'
        max_page_idx_list = []
        for i in range(13):
            cat_list = ['0'] * 13
            cat_list[i] = '1'
            dos, execution, overflow, memc, sqli, xss, dirtra, httprs, bypass, infor, gainpre, csrf, fileinc = cat_list
            page_num = 1
            link = 'https://www.cvedetails.com/vulnerability-list.php?vendor_id=0&product_id=0&version_id=0&page=' + str(
                page_num) + '&hasexp=0&opdos=' + dos + '&opec=' + execution + '&opov=' + overflow + '&opcsrf=' + csrf + '&opgpriv=' + gainpre + '&opsqli=' + sqli + '&opxss=' + xss + '&opdirt=' + dirtra + '&opmemc=' + memc + '&ophttprs=' + httprs + '&opbyp=' + bypass + '&opfileinc=' + fileinc + '&opginf=' + infor + '&cvssscoremin=0&cvssscoremax=0&year=0&month=0&cweid=0&order=1&trc=28068&sha=0ea5fbc52190c28f2a1c51aca205b315bc4c6509'
            page = requests.get(link, timeout=60, headers={'User-Agent': "Magic Browser"})
            print(link)
            # print(dos, ec, ov, csrf, gpriv, sqli, xss, dirt, memc, httprs, byp, fileinc, inf)
            content = BeautifulSoup(page.content).get_text()

            keyword_section = content.replace('\n', ' ')

            loc_1 = keyword_section.find('This Page)')
            loc_2 = keyword_section.find('How does it work? ')
            max_page_idx = keyword_section[loc_1 + 10:loc_2].split('   	')[0].strip().split()[-1]
            print(max_page_idx)
            max_page_idx_list.append(max_page_idx)

        name_cat = ['dos', 'execution', 'overflow', 'memc', 'sqli', 'xss', 'dirtra', 'httprs',  'bypass', 'infor', 'gainpre', 'csrf',  'fileinc' ]
        sha_value_cat = ['38745b427397c23f6ed92e0ed2d3e114da828672',
                     '0ea5fbc52190c28f2a1c51aca205b315bc4c6509',
                     '363372bbc3616054065946a39f4fa589eb5f0f04',
                     '5829c45b747ab5143004640f312c7f72e5b102db',
                     '1b24fccb15090079e49c0131be821c96dc2f001c',
                     'e3bb5586965f5a13bfaa78233a10ebc3f9606d12',
                     '69098b0b30799b9520bf468c7bc060a7f756abf9',
                     'd5623136f5150876a7dfba54b38fc96fe135993c',
                     '7c71486574161a851e392e2e9dcdfea2cde521c3',
                     '1f368a2d3fc25689cc46e4dcb206b4d6103aaab7',
                     '2f1f77e26ecf09cf8b4f251b1efc2b4bcad02050',
                     'e2c3963a5b4ac67dc5dc9fe39ff95f846162e52d',
                     '4160b1b268ed8bcd97bdd927802ef4922995d3d2']
        CVE_id_list_by_cat = []
        try:
            for cat_idx in range(13)[1:]:
                cat_list = ['0'] * 13
                cat_list[cat_idx] = '1'
                sha_value = sha_value_cat[cat_idx]
                dos, execution, overflow, memc, sqli, xss, dirtra, httprs, bypass, infor, gainpre, csrf, fileinc = cat_list
                # cat_list[cat_idx] = '1'
                max_page_num = int(max_page_idx_list[cat_idx])
                print('crawling the CVE ids in the ' + str(cat_idx) + ' category...')
                CVE_id_list_this_cat = []

                page_num = 1
                cve_cnt = 0
                while page_num <= max_page_num:
                    link = 'https://www.cvedetails.com/vulnerability-list.php?vendor_id=0&product_id=0&version_id=0&page=' + str(
                        page_num) + '&hasexp=0&opdos=' + dos + '&opec=' + execution + '&opov=' + overflow + '&opcsrf=' + csrf + '&opgpriv=' + gainpre + '&opsqli=' + sqli + '&opxss=' + xss + '&opdirt=' + dirtra + '&opmemc=' + memc + '&ophttprs=' + httprs + '&opbyp=' + bypass + '&opfileinc=' + fileinc + '&opginf=' + infor + '&cvssscoremin=0&cvssscoremax=0&year=0&month=0&cweid=0&order=1&trc=28068&sha=' + sha_value
                    page = requests.get(link, timeout=60, headers={'User-Agent': "Magic Browser"})
                    print('category ' + str(cat_idx) + ', page ' + str(page_num) + ', cve count ' + str(cve_cnt), link)
                    content = BeautifulSoup(page.content).get_text()
                    content_lines_list = content.split('\n')
                    for line in content_lines_list:
                        if line.startswith('CVE-'):
                            CVE_id_list_this_cat.append(line.strip())
                            cve_cnt += 1
                    page_num += 1
                CVE_id_list_by_cat.append(CVE_id_list_by_cat)

                # f_cve_id_cat_file = open('cve_id/cve_id.' + name_cat[cat_idx], 'w')
                f_cve_id_cat_file = open(self.cve_id_dir + name_cat[cat_idx], 'w')

                idx = 1
                for cve in CVE_id_list_this_cat:
                    f_cve_id_cat_file.write(str(idx) + '\t' + cve + '\n')
                    idx += 1

            print(CVE_id_list_by_cat)

        except requests.exceptions.HTTPError as errh:
            print("Http Error: " + str(errh) + " Please check: " + link)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:" + str(errc) + " Please check: " + link)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:" + str(errt) + " Please check: " + link)
        except requests.exceptions.RequestException as err:
            print("Other errors!" + str(err) + " Please check: " + link)

    def crawl_cve_refs(self):
        files = os.listdir(self.cve_id_dir)
        args = []
        for f in files:
            args.append((f, self.cve_id_dir, self.cve_ref_dir))
            self.pool.map_async(crawl_ref_for_one_vul_category, args)

    def crawl_reports_by_refs(self):
        dict_to_write = dict()
        ref_files = os.listdir(self.cve_ref_dir)
        args = []
        with utils.add_path(self.cve_ref_dir):
            for each_file in ref_files:
                category_module = __import__(each_file.replace('.py', ''))
                cve_ref_dict = category_module.cve_ref_dict
                for cve_id in cve_ref_dict:
                    args.append((cve_id, cve_ref_dict[cve_id], dict_to_write))
        self.pool.map_async(crawl_report, args)
        with open(self.data_dir + 'dataset.py', 'w') as f_write:
            f_write.write('version_dict = ' + str(dict_to_write))


if __name__ == '__main__':

    cveid_dir = 'CVE_ID_DIR'        # specify the path where the collected
                                    # CVE IDs are located.

    cveref_dir = 'CVE_REF_DIR'      # specify the path where the collected report
                                    # hyperlinks referenced by CVE are located.

    dataset_dir = 'DATASET_DIR'     # specify the path where the collected
                                    # reports are located.

    crawler = Crawler(cveid_dir, cveref_dir, dataset_dir, num_of_CPUs=5)
    crawler.run()







