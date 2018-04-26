#include <iostream>
#include "Vocabulary.cpp"
//#define IRIS

using namespace std;
using namespace cuBoW;

int main()
{
#ifdef IRIS
    fstream file("iris.csv");
    string str;
    vector<VectorXf> descriptors;
    while (getline(file, str))
    {
        string::size_type j = 0;
        VectorXf vec(4);
        int num = 0;
        for (string::size_type i = 0; i < str.size(); i++)
        {
            if (str[i] == ',')
            {
                string value = str.substr(j, i - j);
				j = i + 1;
                vec(num) = stod(value);
                num++;
            }
        }
        descriptors.push_back(vec);
    }
    //cout << descriptors.size() << endl;
    vector<vector<VectorXf> > training_features;
    for (int i = 0; i < 5; i++)
    {
        vector<VectorXf> temp;
        for (int j = 0; j < 30; j++)
        {
            temp.push_back(descriptors[i * 30 + j]);
        }
        training_features.push_back(temp);
    }
    //random_shuffle(descriptors.begin(), descriptors.end());
    Vocabulary vocabulary(3, 6);
    vocabulary.create(training_features);
    //cout << 
    auto words = vocabulary.getWords();
    VectorXf sample_word = (*words[5]).descriptor;
    int id = vocabulary.DBfindWord(sample_word);
    int size = sample_word.rows();
    float* dp = (float*)malloc(sizeof(float) * size);
    /*for (int i = 0; i < node_num; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%f ", descriptor_map[i * size + j]);
        }
        printf("\n");
    }*/
    for (int i = 0; i < size; i++)
    {
        dp[i] = sample_word[i];
    }
    int cuid = cudaFindWord(dp, size);
    cout << "id: " << id << endl;
    cout << "cuid: "<< cuid << endl;
    /*vector<Vocabulary::Node> nodes = vocabulary.getVocabulary();
    for (int i = 0 ; i < nodes.size(); i++)
    {
        std::cout << nodes[i].descriptor.transpose() << std::endl;
        int len = nodes[i].descriptor.rows();
        for (int j = 0; j < len; j++)
        {
            cout << descriptor_map[i * len + j] << " ";
        }
        cout << endl << endl;
    }*/
    /*for (size_t i = 0; i < nodes.size(); i++)
    {
        vector<NodeId> children = nodes[i].children;
        cout << i << ": ";
        if(nodes[i].isLeaf())
        {
            cout << "is leaf" << endl;
            continue;
        }
        for (size_t j = 0; j < children.size(); j++)
            cout << children[j] << " ";
        cout << endl;
    }*/
    //vector<Vocabulary::Node*> words = vocabulary.getWords();
    //for (auto word : words)
    //{
        //cout << word->weight << " ";
    //}
#endif
}