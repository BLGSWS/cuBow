#include <fstream>
#include <set>

namespace cuBoW{

class Param_reader
{
public:
    /**
     * 读取参数
     * param fp: 参数文本文件位置
     * */
    Param_reader(const std::string &fp)
    {
        std::ifstream fin;
        fin.open(fp, std::ios::in);
        if (!fin)
        {
            std::cout << "Param: there is no parament file" << std::endl;
            return;
        }
        else
        {}

        while (!fin.eof())
        {   
            std::string str;
            getline(fin, str);

            if (str[0] == '#' || str[0] == '[') continue;
            else
            {}

            int pos = str.find('=');
            if(pos == -1) continue;
            else
            {}

            std::string key = str.substr(0, pos);
            std::string value = str.substr(pos + 1, str.length());

            /// 去空格
            trim(key);
            trim(value);

            params[key] = value;

            if (!fin.good()) break;
            else
            {}
        }
    }
    /**
     * 返回参数值
     * param key: 参数变量名
     * return: 参数值
     * */
    template<class T> 
    T get_param(const std::string &key) const
    {
        std::stringstream ss;
        std::map<std::string, std::string>::const_iterator it = params.find(key);
        if (it == params.end())
        {
            std::cerr << "Params: can not find " << key << " in paraments file" << std::endl;
            throw std::exception();
        }
        ss << it->second;
        T parament;
        ss >> parament;
        return parament;
    }
protected:
    void trim(std::string &str)
    {
        if (!str.empty())
        {
            str.erase(0, str.find_first_not_of(" "));
            str.erase(str.find_last_not_of(" ") + 1);
        }
    }
private:
    Param_reader();
    std::string file_path;
    std::map<std::string, std::string> params;
};

}