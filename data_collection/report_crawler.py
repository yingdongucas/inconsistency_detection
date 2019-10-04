from bs4 import BeautifulSoup
from lxml import html
import requests
import time
from utils_DATA import encode_content, regex_cve, get_cve_id_list
from config import report_list


def crawl_report(args):
    cve_id, ref_link_list, dict_to_write = args
    for ref_link in ref_link_list:
        # if ref_link != 'http://www.securitytracker.com/id/1041303':
        #     continue
        report_category = get_report_category(ref_link)
        if report_category == -1:
            continue
        cve_link = 'https://cve.mitre.org/cgi-bin/cvename.cgi?name=' + cve_id
        target_content_dic = dict()
        if report_category == 1:
            target_content_dic = {'cve_id': [cve_id], 'title': '',
                                  'content': dict_to_write[cve_id]['cve'][cve_link]['content']}
        else:
            target_content_dic = obtain_report_type_and_crawl(ref_link, report_category)
        if target_content_dic in [dict(), None]:
            continue

        add_to_report_dic(dict_to_write, target_content_dic['cve_id'], report_category, ref_link,
                          target_content_dic['title'], target_content_dic['content'])


def add_to_report_dic(dict_to_write, cve_id_list, report_category, report_link, title, content):
    for cve_id in cve_id_list:
        if cve_id not in dict_to_write:
            dict_to_write[cve_id] = {}
        db = report_list[report_category]
        dict_to_write[cve_id][db] = {report_link: {'content': content, 'title': title}}


def get_report_category(ref_link):
    report_category = -1
    if ref_link.find('cve.mitre.org/cgi-bin/cvename.cgi?name') != -1:
        report_category = 1
    elif ref_link.find('securityfocus.com/bid') != -1:
        report_category = 2
    elif ref_link.find('securitytracker.com/id') != -1:
        report_category = 3
    elif ref_link.find('www.securityfocus.com/archiv') != -1:
        report_category = 4
    elif ref_link.find('exploit-db.com') != -1:
        report_category = 5
    elif ref_link.find('openwall.com/lists/oss-security/') != -1:
        report_category = 6
    return report_category


def obtain_report_type_and_crawl(report_link, report_category):
    print('crawling ' + report_link)
    target_content_dic = dict()
    crawl_result = crawl_content_from_link(report_link)
    if crawl_result is None:
        return target_content_dic
    raw_content, clean_content = crawl_result
    # print(clean_content)
    if raw_content == '' or clean_content == '':
        return target_content_dic
    if report_category == 2:
        target_content_dic = parse_securityfocus_official(raw_content)
    elif report_category == 3:
        target_content_dic = parse_securitytracker(raw_content, clean_content)
    elif report_category == 4:
        target_content_dic = parse_securityfocus_forum(raw_content, clean_content)
    elif report_category == 5:
        target_content_dic = parse_edb(raw_content, report_link)
    elif report_category == 6:
        target_content_dic = parse_openwall(raw_content, clean_content)
    else:
        print('ERROR in crawl_report!')
    # print(report_category, target_content_dic)
    return target_content_dic


def crawl_content_from_link(link):
    page = ''
    idx = 0
    while page == '':
        if idx > 10:
            return
        try:
            idx += 1
            page = requests.get(link, timeout=60, headers={'User-Agent': "Magic Browser"})
            break
        except requests.exceptions.HTTPError as errh:
            print("Http Error: " + str(errh) + " Please check: " + link)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:" + str(errc) + " Please check: " + link)
            time.sleep(5)
            print(idx, "Was a nice sleep, now let me continue...")
            continue
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:" + str(errt) + " Please check: " + link)
        except requests.exceptions.RequestException as err:
            print("Other errors!" + str(err) + " Please check: " + link)
    return page.content, BeautifulSoup(page.content).get_text()


def parse_securityfocus_official(raw_content):
    tree = html.fromstring(raw_content)
    keyword_section = tree.xpath('//*[@id="vulnerability"]/span//text()')
    str_content = ''
    if len(keyword_section) > 0:
        keyword_section = keyword_section[0]
        for i in tree.xpath('//*[@id="vulnerability"]/table/tr[9]/td[2]/text()'):
            str_content += i
        cve_id = tree.xpath('//*[@id="vulnerability"]/table/tr[3]/td[2]/text()')
        cve_id_list = []
        for line in cve_id:
            matched_cve = regex_cve(line)
            if len(matched_cve) > 0:
                cve_id_list.append(matched_cve[0])
        return {'cve_id': cve_id_list, 'title': encode_content(keyword_section), 'content': encode_content(str_content)}
    print('ERROR in parse_securityfocus_official')
    return {}


def parse_securitytracker(raw_content, clean_content):
    content = str(raw_content)
    # print(clean_content)
    keyword_section = content[content.find('<title>') + 7: content.find('</title>') - 18].replace('\n', ' ')

    if len(keyword_section) > 0:
        cve_id_list = []
        start_loc = clean_content.find('CVE Reference:')
        end_loc = clean_content.find('(Links to External Site)')
        if start_loc < end_loc and start_loc != -1 and end_loc != -1:
            str_content = clean_content[start_loc:end_loc]
            matched_cve = regex_cve(str_content)
            if len(matched_cve) > 0:
                cve_id_list.append(matched_cve[0])

        start_loc = clean_content.find('Version(s):')
        end_loc = clean_content.find('Description:')
        if start_loc < end_loc and start_loc != -1 and end_loc != -1:
            str_content = clean_content[start_loc + len('Version(s):') :end_loc]
            return {'cve_id': cve_id_list, 'title': encode_content(keyword_section), 'content': encode_content(str_content)}

    print('ERROR in parse_securitytracker')
    return {}


def parse_securityfocus_forum(raw_content, clean_content):
    tree = html.fromstring(raw_content)
    keyword_section = tree.xpath('//*[@id="comments"]/div/a/text()')
    if len(keyword_section) > 0:
        title = keyword_section[0].replace('\n', ' ')
        str_content = ''
        for i in tree.xpath('//*[@id="comments"]/div/div/text()'):
            str_content += i.replace('\n', ' ')
        return {'cve_id': get_cve_id_list(title, clean_content), 'title': encode_content(title), 'content': encode_content(str_content)}
    print('ERROR in parse_securityfocus_forum')
    return {}


def parse_edb(raw_content, report_link):
    tree = html.fromstring(raw_content)
    keyword_section = tree.xpath('/html/body/div/div[2]/div[2]/div/div/div[1]/div/div[1]/h1/text()')
    if len(keyword_section) > 0:
        title = keyword_section[0].replace('\n', ' ')
        cve_line = tree.xpath('/html/body/div/div[2]/div[2]/div/div/div[1]/div/div[2]/div[1]/div[1]/div/div[1]/div/div/div/div[2]/h6/a/text()')

        link = report_link.replace('/exploits/', '/raw/')
        page = requests.get(link, timeout=60, headers={'User-Agent': "Magic Browser"})

        crawl_result = crawl_content_from_link(link)
        if crawl_result is None:
            return {}
        str_content, _ = crawl_result

        str_content = str(page.content).replace('\r\n', ' ')

        if len(cve_line) > 0:
            cve_line = encode_content(cve_line[0])
            for i in cve_line:
                if not (i.isdigit() or i == '-'):
                    cve_line = ''
                    break
            return {'cve_id': ['CVE-' + cve_line], 'title': encode_content(title), 'content': encode_content(str_content)}
        else:
            return {'cve_id': [], 'title': encode_content(title), 'content': encode_content(str_content)}
    print('ERROR in parse_edb')
    return {}


def parse_openwall(raw_content, clean_content):
    tree = html.fromstring(raw_content)
    msg = tree.xpath('/html/body/pre/text()')
    if len(msg) >= 1:
        msg = msg[0]
        loc_start = msg.find('Subject:')
        if loc_start != -1:
            loc_end = msg.find('\n\n', loc_start)
            title = msg[loc_start: loc_end].replace('\n', ' ')
            str_content = ''
            for i in tree.xpath('/html/body/pre/text()'):
                str_content += i.replace('>\n', ' ').replace('\n', ' ')
            return {'cve_id': get_cve_id_list(title, clean_content), 'title': encode_content(title), 'content': encode_content(str_content)}
    print('ERROR in parse_openwall')
    return {}
