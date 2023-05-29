import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # 数据获取与预处理
    with open("anonymous-msweb.data", "r", encoding="utf-8") as f:
        lines = f.readlines()

        # attr_lines = []
        attr_dic = {}
        case_dic = {}
        vote_list = []
        # case_vote_list = []

        valid_attr = ["A", "C", "V"]
        for i, line in enumerate(lines):
            content = line.strip().split(",")

            attr = content[0]
            if attr not in valid_attr:
                continue

            if attr == "A":
                # item = [content[1], content[3], content[4]]
                # attr_lines.append(item)
                attr_dic[content[1]] = [content[3], content[4]]
            elif attr == "C":
                # C要连着处理之后的V
                cur_vote = []
                num = i+1
                if num >= len(lines):
                    break
                while lines[num].strip().split(",")[0] == 'V':
                    cur_vote.append(lines[num].strip().split(",")[1])
                    # case_vote_list.append([content[1], lines[num].strip().split(",")[1]])
                    num += 1
                    if num >= len(lines):
                        break
                vote_list.append(cur_vote)
                case_dic[content[1]] = cur_vote[1:]
            else:
                continue

        print(1)

    # 频繁模式挖掘
    # 转换数据集,在使用apriori函数时，输入的数据集需要进行一些预处理才能正确执行。具体来说，需要将数据集转换为适合apriori函数的形式，可以使用TransactionEncoder类来完成转换。
    te = TransactionEncoder()
    # 将每一行case_vote数据转换为一个[1, 285]的向量，其中285是在全部数据中出现的不同attr的总数，如果这行case_vote数据中有某一个attr，那么对应位置为True，否则为False
    # te_ary总共为[32711,285]，32711为case_dic的总数，也就是所有case的总数
    te_ary = te.fit_transform(vote_list)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 使用apriori函数进行频繁模式挖掘
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

    print(frequent_itemsets)
    print(1)

    # 导出关联规则
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    print(rules)

    # 计算支持度和置信度
    rules['support'] = rules['support'].round(4)
    rules['confidence'] = rules['confidence'].round(4)

    # 评价规则：使用Lift和卡方指标
    rules['lift'] = rules['support'] / rules['antecedent support']
    rules['chi_square'] = (df.shape[0] * (rules['support'] - rules['antecedent support'] * rules['consequent support']) ** 2) \
                          / (rules['antecedent support'] * rules['consequent support'] * (1 - rules['antecedent support']) * (1 - rules['consequent support']))

    # 对挖掘结果进行分析
    sorted_rules = rules.sort_values(by='lift', ascending=False)

    print("关联规则挖掘结果：")
    print(sorted_rules)

    print(1)

    # 可视化展示
    # 可视化频繁项集支持度
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x='support', y='itemsets', data=frequent_itemsets.sort_values(by='support', ascending=False),
                orient='h')
    plt.title('Frequent Itemsets - Support')
    plt.xlabel('Support')
    plt.ylabel('Itemsets')

    plt.tight_layout()
    plt.show()

    # 可视化关联规则评估指标
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='lift', y='confidence', data=rules)
    plt.title('Association Rules - Lift vs. Confidence')
    plt.xlabel('Lift')
    plt.ylabel('Confidence')
    plt.show()
