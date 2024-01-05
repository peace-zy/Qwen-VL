#-*-coding: utf-8 -*-
"""
Author:   zhangyan75@baidu.com
Date  :   21/12/08 10:34:34
Desc  :   
"""

import sys
import os
import traceback
import json
import re

def make_path(path):
    """创建路径"""
    if not os.path.exists(path):
        os.makedirs(path)

def rm_file(f):
    """删除文件"""
    if os.path.exists(f):
        os.remove(f)

def txt2html(gt_file, res_file):
    """txt2html"""
    html_file = os.path.basename(res_file).split('.')[0] + '.html'
    rm_file(html_file)

    js_part = '''
<body>
    <div id="div_top" style="top:0px;left:50px;width:100%;height: 30px;position:absolute;">
        <button onclick="Count()">汇总</button>
        <button onclick="ExportData()">导出数据</button>
    </div>
</body>

<!--jspart-->
<script language="javascript" type="text/javascript">
    function update(sender) {
        console.log(sender);
        sender.defaultChecked = !sender.defaultChecked;
        sender.checked = sender.defaultChecked;
    }

    function Count() {
        var num_image = 0
        var num_region = 0
        var num_right = 0
        var num_wrong = 0
        var num_hold = 0
        var show_table =document.getElementById("show_table");
        for (var i = 1; i < show_table.rows.length; ++i) {
            num_image += 1
            var cells = show_table.rows[i].cells
            for (var j = 0; j < cells.length; ++j) {
                num_region += 1
                radios = cells[j].getElementsByTagName('input')
                for (var k = 0; k < radios.length; ++k) {
                    if (radios[k].checked) {
                        if (radios[k].value == 'Right') {
                            num_right += 1
                        }
                        else if (radios[k].value == 'Wrong') {
                            num_wrong += 1
                        }
                        else if (radios[k].value == 'Hold') {
                            num_hold += 1
                        }
                    }
                }
            }
        }
        alert("图像数量=" + num_region  + ", 对的数量=" + num_right + ", 错的数量=" + num_wrong + ", 待定数量=" + num_hold + ", 未标注数量=" + (num_region - num_right - num_wrong - num_hold) + ", 精确率=" + Number(100 * num_right / num_region).toFixed(2) + "%")
         
    }

    function download(filename, text) {
        var element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
        element.setAttribute('download', filename);
        element.click();
    }
     
    function ExportData() {
        var right_line = ""
        var wrong_line = ""
        var show_table =document.getElementById("show_table");
        for (var i = 1; i < show_table.rows.length; ++i) {
            var cells = show_table.rows[i].cells
            for (var j = 0; j < cells.length; ++j) {
                img_src = cells[j].getElementsByTagName("img")[0].src
                radios = cells[j].getElementsByTagName('input')
                for (var k = 0; k < radios.length; ++k) {
                    if (radios[k].checked) {
                        if (radios[k].value == 'Right') {
                            right_line += img_src + "\\n"
                        }
                        else if (radios[k].value == 'Wrong') {
                            wrong_line += img_src + "\\n"
                        }
                    }
                }
            }
        }
        console.log("right")
        download('right.txt', right_line)
        console.log("wrong")
        download('wrong.txt', wrong_line)
        alert("导出数据成功")

    }


</script>
'''
    
    wlines = []
    wlines.append(js_part)
    wlines.append('<table border=1 id="show_table" style="top:30px;left:0px;position:absolute;word-break:break-all">\n')
    wlines.append('<tr><td>改造前</td><td>改造后</td><td>居住需求</td><td>设计理念</td><td>生成结果</td></tr>\n')
    
    COL_NUMS = 1
    id_r = 0
    cnt = 0
    with open(res_file, "r") as f:
        res_data = json.load(f)
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    pre_fix = "http://localhost:8000/data/nfs-ten9/nfs/zhangyan461"
    for idx, ele in enumerate(gt_data):
        image_id = ele["id"]
        conversations = ele["conversations"]
        info = conversations[0]["value"].split("\n图1是原始户型图，图2是改造后的户型图，居住需求如下\n")
        image_info = re.split("<qwen_vl_img>|</qwen_vl_img>", info[0])
        before_img_url, after_img_url = os.path.join(pre_fix, image_info[1]), os.path.join(pre_fix, image_info[3])
        demand = info[1].split("\n帮我生成一下改造户型的设计理念")[0]
        idea = conversations[1]["value"]
        predict = None
        for item in res_data:
            if item["image_id"] == image_id:
                predict = item["caption"]
                break
        if predict is None:
            print("predict is None")

        if cnt == 0:
            wl = '<tr>' 
            #print(cnt, 's')
        c = 0
        select = """<label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                    <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Right" onclick="update(this)" />对</label>
                    <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                    <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Wrong" onclick="update(this)" />错</label>
                    <label class="radio-inline" style="font-family: 'Microsoft YaHei UI';font-size: medium;">
                    <input type="radio" display:block name="result{}" id="optionsRadios{}" value="Hold" onclick="update(this)" />待定</label>""".\
                    format(id_r, c, id_r, c + 1, id_r, c + 2)
        c += 3
        id_r += 3

        wl = '<td><a href="{}" src="{}"> '\
                '<img src="{}" width=216 border=1 controls></a></td>'.\
                format(before_img_url, before_img_url, before_img_url)
        wl += '<td><a href="{}" src="{}"> '\
                '<img src="{}" width=216 border=1 controls></a></td>'.\
                format(after_img_url, after_img_url, after_img_url)
        wl += '<td>{}</td><td>{} width="50%"</td><td>{}<br />{}</td>'.\
                format(demand, idea, predict, select)
        cnt += 1

        if cnt == COL_NUMS:
            wl += '</tr>\n'    
            #print(cnt, 'e')
            cnt = 0
        wlines.append(wl)
  
    if cnt and cnt < COL_NUMS:
        wlines.append('</tr>\n')

    wlines.append('\n</table>')
    with open(html_file, 'w') as file:
        file.writelines(wlines)

def main():
    """main"""
    """python gen_html_for_image_url.py [infile]"""
    gt_file = sys.argv[1]
    res_file = sys.argv[2]

    txt2html(gt_file, res_file)
    return

if __name__ == "__main__":
    main()

