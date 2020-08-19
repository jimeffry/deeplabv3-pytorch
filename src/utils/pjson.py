import json
import numpy as np
import tqdm 

def processJson(jsonfile,outfile,outfile2):
    fr = open(jsonfile,'r')
    fw = open(outfile,'w')
    fw2 = open(outfile2,'w')
    fcnts = json.load(fr)
    datalists = fcnts['data']['results']
    total_num = int(fcnts['data']['total'])
    # idst = '124.152.117.60'
    for i in tqdm.tqdm(range(total_num)):
        tmp = datalists[i]
        # if int(tmp['alive']) == 1:
        #     # fw.write("{}\t{}\t{}\n".format(tmp['name'],tmp['gbID'],tmp['rtspUrl']))
        #     fw.write("{}\n".format(tmp['rtspUrl']))
        tmp_url = tmp['rtspUrl'].strip()
        if len(tmp_url)>0:
            # tmp_s = tmp_url.split('//')
            # tmp_id_s = tmp_s[1].split(':')
            # id_url = tmp_s[0]+'//'+idst+':'+tmp_id_s[1]
            fw.write("{}\t{}\n".format(tmp['name'],tmp_url))
            id_url = tmp_url+'/livestream'
            fw2.write("{}\n".format(id_url))
    fr.close()
    fw.close()
    fw2.close()

if __name__=='__main__':
    infile = '/data/wuwei.json'
    infile = '/data/response.json'
    outfile = '../datasets/wuwei_namertl.txt'
    rtlfile = '../datasets/wuwei_rtl.txt'
    processJson(infile,outfile,rtlfile)