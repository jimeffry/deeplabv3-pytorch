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

def processTxt(txtfile,outfile1,outfile2):
    '''
    txtfile: siPing rtl files
    '''
    fr = open(txtfile,'r')
    fw1 = open(outfile1,'w')
    fw2 = open(outfile2,'w')
    fcnts = fr.readlines()
    # fcnts2 = fw1.readlines()
    total = len(fcnts)
    # print(fcnts[1].strip().split())
    tmplist = []
    for i,tmp in enumerate(fcnts):
        if (i+1)%3==0  :
            # print(tmplist)
            fw2.write(tmplist[0]+tmplist[1]+'\n')
            fw1.write(tmplist[1]+'\n')
            tmplist = []
        if len(tmp.strip().split())>0:
            tmplist.append(tmp.strip())
            if i<7:
                print(i,tmp)
                print(tmplist)
    '''
    for i in range(total):
        tmp = fcnts[i].strip()
        tmp_sp = tmp.split()
        tmpws = tmp_sp[1].split(':')
        fw2.write(fcnts2[i].strip()+'\t'+tmpws[1]+'\n')
    '''
    fr.close()
    fw1.close()
    fw2.close()


if __name__=='__main__':
    infile = '/data/wuwei.json'
    infile = '/data/response.json'
    outfile = '../datasets/wuwei_namertl.txt'
    rtlfile = '../datasets/wuwei_rtl.txt'
    # processJson(infile,outfile,rtlfile)
    infile = '/data/siping.txt'
    outfile = '../datasets/siping_namertl.txt'
    rtlfile = '../datasets/siping_rtl.txt'
    # outfile = '/data/siping_new2.txt'
    infile = '/data/siping_new2.txt'
    infile2 = '/data/siping_new.txt'
    infile = '/data/rtsp_url_kd.txt'
    outfile = '/data/siping_80.txt'
    outfile1 = '../datasets/siping_rtl_80.txt'
    processTxt(infile,outfile1,outfile)