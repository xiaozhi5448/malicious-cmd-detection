import requests
from bs4 import BeautifulSoup
import logging
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(filename)s %(lineno)d: %(levelname)s  %(message)s')
output_dir = '../../command_linuxde'
root_url = 'https://man.linuxde.net/'

results = {}


def get_command_links():
    data = {}
    homepage = requests.get(root_url)
    homepage.encoding = 'utf-8'
    if homepage.status_code != 200:
        logging.error("visit homepage failed!")
        return None
    home_content = BeautifulSoup(homepage.content, "html.parser")
    tags_list_block = home_content.select("#tags-list")[0]
    tags_dl_list = tags_list_block.select("dl")
    for tag_dl in tags_dl_list:
        title = tag_dl.select('dt')[0].text
        logging.info("processing catagory: {}".format(title))
        if title not in data:
            data[title] = {}
        results[title] = {}
        tags_dd_list = tag_dl.select("dd")
        for tag_dd in tags_dd_list:
            sub_link = tag_dd.a['href']

            sublist_content = requests.get(sub_link)
            if sublist_content.status_code != 200:
                logging.error("error occured when get sub list of {}".format(title))
                continue
            sublist_content.encoding = 'utf-8'
            sub_soup = BeautifulSoup(sublist_content.content, "html.parser")
            command_links = sub_soup.select("ul#arcs-list > li > a")
            for command_item in command_links:

                command = command_item['title']
                command_link = command_item['href']
                data[title][command] = command_link
                results[title][command] = []
                logging.info("find command: {}; link:{}".format(command, command_link))
    return data

def get_command_content(catagory, command, url):
    logging.info("process command {} with url:{}".format(command, url))
    page = requests.get(url)
    page.encoding = 'utf-8'
    if page.status_code != 200:
        logging.error("error occured when getting content of {}".format(url))
        return None
    dom_root = BeautifulSoup(page.content, "html.parser")
    dom_arc = dom_root.select("div#arc-body")[0]
    headers = dom_arc.select("h3")
    headers.extend(dom_arc.select("h2"))
    header = None
    for item in headers:
        if item.text.strip() in ["实例", "例子"] :
            header = item
            break
    if not header:
        logging.info("command {} has no instance!".format(command))
        return
    tag_instance = header
    tag_pres = tag_instance.find_next_siblings("pre")
    commands = []
    for tag_pre in tag_pres:
        content = [line.strip() for line in  tag_pre.text.split('\n')]
        commands.extend(content)
    logging.info("command {} parse finished, {} command found".format(command, len(commands)))
    return catagory, command, commands

def main_loop():
    command_links = {}
    if os.path.exists('links.bin'):
        with open('links.bin', 'rb') as infp:
            command_links = pickle.load(infp)
    else:
        command_links = get_command_links()
        with open('links.bin', 'wb') as outfp:
            pickle.dump(command_links, outfp)
    executor = ThreadPoolExecutor(max_workers=10)
    tasks = []
    for catagory in command_links:
        for command in command_links[catagory]:
            task_ref = executor.submit(get_command_content, catagory, command, command_links[catagory][command])
            tasks.append(task_ref)
    for task_ref in as_completed(tasks):
        data = task_ref.result()
        if data:
            if data[0] not in results:
                results[data[0]] = {}
            if data[1] not in results[data[0]]:
                results[data[0]][data[1]] = []
            results[data[0]][data[1]].extend(data[2])

def save_command():
    logging.info('dumping data to disk...')
    for category in results:
        category_name = category
        if '|' in category:
            category_name = ''.join(category.split(' | '))

        category_path = os.path.join(output_dir, category_name) + '.txt'
        with open(category_path, 'w', encoding='utf-8') as outfp:
            for command in results[category]:
                outfp.write('\n'.join(results[category][command]))
                outfp.write('\n')
    logging.info('save finished!')


if __name__ == '__main__':
    main_loop()
    save_command()