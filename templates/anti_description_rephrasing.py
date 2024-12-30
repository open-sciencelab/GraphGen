TEMPLATE_EN: str = """-Goal-
Transform the input sentence into its opposite meaning while:

1. Preserving most of the original sentence structure
2. Changing only key words that affect the core meaning
3. Maintaining the same tone and style
4. The input sentence provided is a right description, and the output sentence should be a wrong description
5. The output sentence should be fluent and grammatically correct

################
-Examples-
################
Input:
The bright sunshine made everyone feel energetic and happy.

Output:
The bright sunshine made everyone feel tired and sad.

################
-Real Data-
################
Input: 
{input_sentence}
################
Output:
"""

TEMPLATE_ZH: str = """-目标-
将输入句子转换为相反含义的句子，同时：

1. 保留大部分原始句子结构
2. 仅更改影响核心含义的关键词
3. 保持相同的语气和风格
4. 提供的输入句子是一个正确的描述，输出句子应该是一个错误的描述
5. 输出句子应该流畅且语法正确

################
-示例-
################
输入：
明亮的阳光让每个人都感到充满活力和快乐。

输出：
明亮的阳光让每个人都感到疲惫和悲伤。

################
-真实数据-
################
输入：
{input_sentence}
################
输出：
"""

ANTI_DESCRIPTION_REPHRASING_PROMPT= {
    "English": {
        "TEMPLATE": TEMPLATE_EN
    },
    "Chinese": {
        "TEMPLATE": TEMPLATE_ZH
    }
}
