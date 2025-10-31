#include <bits/stdc++.h>
#define endl "\n"

using namespace std;

// 字符串内容替换。str: 字符串，old_value: 被替换值，new_value: 替换后的新值
void replace_str(string& str, const string& old_value, const string& new_value){
    for(string::size_type pos(0); pos != string::npos; pos += new_value.length()) {
        if((pos = str.find(old_value,pos)) != string::npos)
        str.replace(pos, old_value.length(), new_value);
    else break;
    }
}

// 判断输入的是否是数字串
bool num_judge(int num){
    string s = to_string(num);
    regex reg("[0-9]+");
    if(regex_match(s, reg)) return true;
    else return false;
}

class csv{
public:
    void setPath(string pth);      // 设置路径
    string getPath(){ return path; }  // 取路径
    bool isFileExists(string pth); //判断 csv 文件是否存在
    bool isCSVFile(string pth);    // 判断是否为 csv 文件。注意，其中因为对正则表达式不熟悉，所以对csv文件名合法性匹配的正则表达式不是很完善
    void build(int ln, int cl);    // 建立新 csv
    void rewrite(int ln, int cl);  // 重写本 csv
    void append();                 // 本 csv 文件尾追加新行
    void read();                   // 读取本 csv
    void add(int pos = -1);        // 指定位置添加新行。默认 pos 小于 0 时结尾追加
    void del(int start_pos = -1, int end_pos = 0);  // 删除 start_pos 至 end_pos 的行
private:
    string path;                   // csv 文件路径
    int line, col;                 // 行、列
};

void csv::setPath(string pth){
    if(isCSVFile(pth) == false){
        cout << "打开的不是CSV文件!请重试!(如果是新建，请忽略此条信息)" << endl;
    }else{
        path = pth;
        ifstream f(path, ios::in);
        if(!f){
            cout << "打开" << path << "失败！请重试！" << endl;
        }else{
            int cl = 0, ln = 0;
            string s;
            getline(f, s);
            int len = s.length();
            for(int i = 0; i < len; i++){
                if(s[i] == ',') cl++;
            }
            cl++;
            while(getline(f, s)){
                ln++;
            }
            ln++;
            col = cl;
            line = ln;
            f.close();
            cout << "打开文件" << getPath() << "成功！" << endl;
        }
    }
}

bool csv::isFileExists(string pth){
    ifstream f(pth.c_str());
    return f.good();
}

bool csv::isCSVFile(string pth){
    regex reg(".+\.csv");
    if(regex_match(pth, reg)) return true;
    else return false;
}

void csv::build(int ln, int cl){
    if(isFileExists(path)){
        cout << "文件" << path << "已存在！" << endl;
    }else{
        ofstream f(path, ios::out);
        line = ln;
        col = cl;
        cout << "文件" << path << "创建成功！" << endl;
        f.close();
    }
}

void csv::rewrite(int ln, int cl){
    if(isCSVFile(path) == false){
        cout << "打开 / 创建的不是CSV文件!请重试!" << endl;
    }else{
        ofstream f(path, ios::out);
        if(!f){
            cout << "打开" << path << "失败！请重试！" << endl;
        }else{
            line = ln;
            col = cl;
            cout << "请输入内容（空格或换行间隔）：" << endl;
            string s;
            for(int i = 0; i < line; i++){
                for(int j = 0; j < col; j++){
                    cin >> s;
                    f << s << ((j != col - 1) ? "," : "");
                }
                f << endl;
            }
            f.close();
            cout << "写入数据完成！" << endl;
        }
    }
}

void csv::append(){
    if(isCSVFile(path) == false){
        cout << "打开的不是CSV文件!请重试!" << endl;
    }else{
        ofstream f(path, ios::out|ios::app);
        if(!f){
            cout << "打开" << path << "失败！请重试！" << endl;
        }else{
            int ln;
            cout << "请输入追加行数：" << endl;
            cin >> ln;
            line += ln;
            cout << "请输入追加内容（空格或换行间隔）：" << endl;
            string s;
            f << endl;
            for(int i = 0; i < ln; i++){
                for(int j = 0; j < col; j++){
                    cin >> s;
                    f << s << ((j != col - 1) ? "," : "");
                }
                f << endl;
            }
            f.close();
            cout << "写入数据完成！" << endl;
        }
    }
}

void csv::read(){
    if(isCSVFile(path) == false){
        cout << "打开的不是CSV文件！请重试！" << endl;
    }else{
        ifstream f(path, ios::in);
        if(!f){
            cout << "打开" << path << "失败！请重试！" << endl;
        }else{
            int pos = 0, ln = 0;
            string s;
            cout << "---------------------------------" << endl;
            while(getline(f, s)){
                ln++;
                while(s.find(',') != std::string::npos){
                    pos = s.find(',');
                    replace_str(s, ",", " ");
                }
                cout << s << endl;
            }
            line = ln;
            f.close();
            cout << "---------------------------------" << endl;
            cout << "共 " << line << " 行，" << col << " 列" << endl;
            cout << "\n读取完毕！" << endl;
        }
    }
}

void csv::add(int pos){
    if(pos < 0){
        append();
    }else{
        if(isCSVFile(path) == false){
            cout << "打开的不是CSV文件！请重试！" << endl;
        }else{
            ifstream f1(path, ios::in);
            if(!f1){
                cout << "打开" << path << "失败！请重试！" << endl;
            }else{
                ofstream f(path);
                int cnt = 1, i_pos = 0;
                string ss;
                while(cnt <= pos){
                    getline(f1, ss);
                    cnt++;
                    i_pos += ss.length();
                }
                f.seekp(i_pos);
                int ln;
                cout << "请输入追加行数：" << endl;
                cin >> ln;
                line += ln;
                cout << "请输入追加内容（空格或换行间隔）：" << endl;
                string s;
                f << endl;
                for(int i = 0; i < ln; i++){
                    for(int j = 0; j < col; j++){
                        cin >> s;
                        f << s << ((j != col - 1) ? "," : "");
                    }
                    f << endl;
                }
                f.close();
                f1.close();
                cout << "写入数据完成！" << endl;
            }
        }
    }
}

void csv::del(int start_pos, int end_pos){
    vector<string> tmp_files;
	ifstream f(path);
	if(!f){
		cout << "打开" << path <<" 失败！请重试！" << endl;
	}else{
	    string lineContent;
	    while(getline(f, lineContent, '\n')){
		    tmp_files.push_back(lineContent + "\n" );
	    }
	    f.close();
	    ofstream of(path, ios::out);
	    vector<string>::iterator siter = tmp_files.begin();
	    if(start_pos < 0){
            copy(tmp_files.begin(), tmp_files.end() - 1, ostream_iterator<string>(of));
            line--;
	    }else{
	        int num = end_pos - start_pos + 1;
	        line -= num;
            while(num--){
                tmp_files.erase(tmp_files.begin() + start_pos - 1);
            }
            copy(tmp_files.begin(), tmp_files.end(), ostream_iterator<string>(of));
	    }
	    cout << "删除成功！" << endl;
	    of.close();
	}
}

// int main(){
//     int num, pos, line, col, end_pos, flag = 1;
//     string path;
//     csv c;
//     while(flag){
//         cout << "---------------------------------" << endl;
//         cout << "<功能选择>" << endl;
//         cout << "0.打开csv文件\t1.新建csv文件\n2.读取当前文件\t3.重写当前文件\n4.在末尾添加指定行新数据\t5.删除最后一行数据\n6.在指定位置添加一行新数据\t7.删除指定行数据" << endl;
//         cout << "8.退出" << endl;
//         cout << "---------------------------------" << endl;
//         cout << "$ ";
//         cin >> num;
//         if(num_judge(num) == false){
//             cout << "输入非法！" << endl;
//         }else{
//             switch(num){
//                 case 0:
//                     cout << "请输入路径：" << endl;
//                     cin >> path;
//                     c.setPath(path);
//                     break;
//                 case 1:
//                     cout << "请输入路径：" << endl;
//                     cin >> path;
//                     c.setPath(path);
//                     cout << "请输入新文件的列数：";
//                     cin >> col;
//                     c.build(0, col);
//                     break;
//                 case 2:
//                     c.read();
//                     break;
//                 case 3:
//                     cout << "输入新文件行数：";
//                     cin >> line;
//                     cout << "输入新文件列数：";
//                     cin >> col;
//                     c.rewrite(line, col);
//                     break;
//                 case 4:
//                     c.append();
//                     break;
//                 case 5:
//                     c.del(-1);
//                     break;
//                 case 6:
//                     cout << "插入位置：";
//                     cin >> pos;
//                     c.add(pos);
//                     break;
//                 case 7:
//                     cout << "删除起始位置：";
//                     cin >> pos;
//                     cout << "删除终止位置：";
//                     cin >> end_pos;
//                     c.del(pos, end_pos);
//                     break;
//                 case 8:
//                     flag = 0;
//                 default:
//                     cout << "输入非法！" << endl;
//                     break;
//             }
//         }
//     }
//     return 0;
// }

