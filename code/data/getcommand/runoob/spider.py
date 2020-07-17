from bs4 import BeautifulSoup
import logging
import requests
import re
import json
import random
import os
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')

root_url = 'https://www.runoob.com{}'
start_point = '/linux/linux-command-manual.html'
output_dir = 'data/command_runoob'
def parse_data():
    main_cata_links = {}
    start_url = root_url.format(start_point)
    res = requests.get(start_url)
    res.encoding = 'utf-8'
    if res.status_code == 200:
        root_dom = BeautifulSoup(res.content, "html.parser")
        table_main = root_dom.select("#content > table.reference")[0]
        tag_trs = table_main.select("tr")
        current_cata_name  = ''
        for tag_tr in tag_trs:
            if tag_tr.select('th'):
                continue
            elif tag_tr.select('strong'):
                cata_name = tag_tr.select("strong")[0].text.strip().split('、')[-1]
                main_cata_links[cata_name] = []
                current_cata_name = cata_name
            else:
                tds = tag_tr.select("td")
                for tag_td in tds:
                    if tag_td.a:
                        command = tag_td.a.text.strip()
                        sub_url = tag_td.a['href']
                        if '/' not in sub_url:
                            sub_url = '/linux/{}'.format(sub_url)
                        main_cata_links[current_cata_name].append((command, sub_url))
    return main_cata_links

def get_real_command(links:dict):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    error_outfp = open('error.txt', 'w', encoding='utf-8')

    total_command_count = 0
    for command_catagory in links:
        data_outfp = open(os.path.join(output_dir, '{}.txt'.format(command_catagory)), 'w', encoding='utf-8')
        for link_item in links[command_catagory]:
            logging.info("processing command: {}".format(link_item[0]))
            url = root_url.format(link_item[1])
            command_name = link_item[0]
            res = requests.get(url)
            res.encoding = 'utf-8'
            if res.status_code == 200:
                tag_root = BeautifulSoup(res.content, "html.parser")
                main_content = tag_root.select("#content")[0]
                if main_content.select("h3"):
                    instance_title = main_content.select("h3")[-1]
                    tag_pres = instance_title.find_next_siblings("pre")
                    logging.info("{} example command about {}".format(len(tag_pres), command_name))
                    total_command_count += len(tag_pres)
                    for tag_pre in tag_pres:
                        command_text = tag_pre.text.strip()
                        command_line = command_text.split('\r\n')[0]


                        items = re.split("[#$]", command_line)
                        # if len(items) == 3:
                        #     command_line = items[1]
                        # elif len(items) == 2:
                        #     if not items[0] or '[' in items[0]:
                        #         command_line = items[1]
                        #     else:
                        #         command_line = items[0]


                        data_outfp.write(command_line + os.linesep)
            else:
                logging.error("error occured when getting content of command: {}\nurl: {}".format(command_name, url))
                error_outfp.write('{}{}'.format(url, os.linesep))
        data_outfp.close()
    logging.info("total command:{}".format(total_command_count))
    error_outfp.close()

def process_comamands():
    filelist = os.listdir(output_dir)
    commands = []
    for filename in filelist:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as infp:
            for line in infp.readlines():
                if line:
                    command = ''
                    items = re.split('[#$><]', line.strip())
                    items = [item for item in items if item != '']
                    if len(items) == 2:
                        if items[0] and '[' in items[0]:
                            command = items[1].strip()
                        else:
                            command = items[0].strip()
                    elif len(items) == 3:
                        command = items[1].strip()
                    elif len(items) == 1:
                        command = items[0].strip()
                    commands.append(command)
    with open(os.path.join(output_dir, 'total.txt'), mode='w', encoding='utf-8') as outfp:
        for command in commands:
            if command:
                outfp.write(command + '\n')

if __name__ == '__main__':
    # cata_links = parse_data()
    # get_real_command(cata_links)
    process_comamands()
