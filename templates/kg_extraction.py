TEMPLATE_EN: str = """You are an NLP expert, skilled at analyzing text to extract named entities and their relationships.

-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

################
-Examples-
################
-Example 1-
Text:
################
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.
"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."
Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."
Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.
The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
################
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}

-Example 2-
Text:
#############
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.
Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.
Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}

################
-Real Data-
################
Entity_types: {entity_types}
Text: {input_text}
################
Output:
"""

# TODO: 修改英文示例

TEMPLATE_ZH: str = """你是一个NLP专家，擅长分析文本提取命名实体和关系。

-目标-
给定一个实体类型列表和可能与列表相关的文本，从文本中识别所有这些类型的实体，以及这些实体之间所有的关系。
使用{language}作为输出语言。

-步骤-
1. 识别所有实体。对于每个识别的实体，提取以下信息：
   - entity_name：实体的名称，首字母大写
   - entity_type：以下类型之一：[{entity_types}]
   - entity_summary：实体的属性与活动的全面总结
   将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_summary>)
   
2. 从步骤1中识别的实体中，识别所有（源实体，目标实体）对，这些实体彼此之间*明显相关*。
   对于每对相关的实体，提取以下信息：
   - source_entity：步骤1中识别的源实体名称
   - target_entity：步骤1中识别的目标实体名称
   - relationship_summary：解释为什么你认为源实体和目标实体彼此相关
   - relationship_strength：一个数值分数，表示源实体和目标实体之间关系的强度
   - relationship_keywords：一个或多个高级关键词，总结关系的主要性质，关注概念或主题而非具体细节
   将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_summary>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别总结整个文本的主要概念、主题或话题的高级关键词。这些应该捕捉文档中存在的总体思想。
   将内容级关键词格式化为("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以中文返回步骤1和2中识别出的所有实体和关系的输出列表。使用**{record_delimiter}**作为列表分隔符。

5. 完成后，输出{completion_delimiter}

################
-示例-
################
-示例 1-
文本：
################
鲁镇的酒店的格局，是和别处不同的：都是当街一个曲尺形的大柜台，柜里面预备着热水，可以随时温酒。做工的人，傍午傍晚散了工，每每花四文铜钱，买一碗酒，——这是二十多年前的事，现在每碗要涨到十文，——靠柜外站着，热热的喝了休息；倘肯多花一文，便可以买一碟盐煮笋，或者茴香豆，做下酒物了，如果出到十几文，那就能买一样荤菜，但这些顾客，多是短衣帮，大抵没有这样阔绰。只有穿长衫的，才踱进店面隔壁的房子里，要酒要菜，慢慢地坐喝。
################
输出：
("entity"{tuple_delimiter}"鲁镇的酒店"{tuple_delimiter}"location"{tuple_delimiter}"鲁镇的酒店是一个特定地点，其格局独特，柜台形状为曲尺形，提供热水温酒服务。"){record_delimiter}
("entity"{tuple_delimiter}"曲尺形的大柜台"{tuple_delimiter}"keyword"{tuple_delimiter}"曲尺形的大柜台是鲁镇酒店内独特的设施，用于提供服务。"){record_delimiter}
("entity"{tuple_delimiter}"热水温酒"{tuple_delimiter}"keyword"{tuple_delimiter}"热水温酒是鲁镇酒店提供的一项服务，顾客可以随时温酒。"){record_delimiter}
("entity"{tuple_delimiter}"做工的人"{tuple_delimiter}"person"{tuple_delimiter}"做工的人是鲁镇酒店的常客，通常在工作结束后花四文铜钱买一碗酒，有时还会买一些下酒菜。"){record_delimiter}
("entity"{tuple_delimiter}"二十多年前的事"{tuple_delimiter}"date"{tuple_delimiter}"二十多年前的事是指过去的时间点，当时一碗酒的价格为四文铜钱。"){record_delimiter}
("entity"{tuple_delimiter}"现在"{tuple_delimiter}"date"{tuple_delimiter}"现在是指当前的时间点，与过去相比，一碗酒的价格涨到了十文。"){record_delimiter}
("entity"{tuple_delimiter}"短衣帮"{tuple_delimiter}"concept"{tuple_delimiter}"短衣帮是指做工的人，他们通常穿着短衣，经济条件有限。"){record_delimiter}
("entity"{tuple_delimiter}"穿长衫的"{tuple_delimiter}"person"{tuple_delimiter}"穿长衫的是鲁镇酒店的另一类顾客，他们经济条件较好，通常会进入店面隔壁的房间慢慢喝酒吃菜。"){record_delimiter}
("entity"{tuple_delimiter}"盐煮笋"{tuple_delimiter}"food"{tuple_delimiter}"盐煮笋是鲁镇酒店提供的一种下酒菜，顾客可以花一文铜钱购买。"){record_delimiter}
("entity"{tuple_delimiter}"茴香豆"{tuple_delimiter}"food"{tuple_delimiter}"茴香豆是鲁镇酒店提供的另一种下酒菜，顾客可以花一文铜钱购买。"){record_delimiter}
("entity"{tuple_delimiter}"荤菜"{tuple_delimiter}"food"{tuple_delimiter}"荤菜是鲁镇酒店提供的较为昂贵的菜品，顾客需要花十几文铜钱购买。"){record_delimiter}
("relationship"{tuple_delimiter}"鲁镇的酒店"{tuple_delimiter}"曲尺形的大柜台"{tuple_delimiter}"鲁镇的酒店内设有一个曲尺形的大柜台，用于提供服务。"{tuple_delimiter}"服务, 能力"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"鲁镇的酒店"{tuple_delimiter}"热水温酒"{tuple_delimiter}"鲁镇的酒店提供热水温酒服务，顾客可以随时温酒。"{tuple_delimiter}"服务, 方便"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"做工的人"{tuple_delimiter}"二十多年前的事"{tuple_delimiter}"做工的人在二十多年前花四文铜钱买一碗酒，反映了当时的生活成本。"{tuple_delimiter}"历史背景, 生活成本"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"做工的人"{tuple_delimiter}"现在"{tuple_delimiter}"现在做工的人需要花十文铜钱买一碗酒，反映了物价的上涨。"{tuple_delimiter}"经济变化, 生活成本"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"做工的人"{tuple_delimiter}"短衣帮"{tuple_delimiter}"做工的人属于短衣帮，通常经济条件有限。"{tuple_delimiter}"社会等级, 经济状态"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"做工的人"{tuple_delimiter}"穿长衫的"{tuple_delimiter}"做工的人与穿长衫的形成对比，反映了社会阶层的差异。"{tuple_delimiter}"社会分层, 经济差距"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"穿长衫的"{tuple_delimiter}"鲁镇的酒店"{tuple_delimiter}"穿长衫的顾客通常会进入鲁镇酒店的房间慢慢喝酒吃菜，享受更高级的服务。"{tuple_delimiter}"服务质量, 经济状况"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"社会分层, 经济差距, 服务, 生活成本, 历史背景"){completion_delimiter}

-示例 2-
文本：
################
摘要 黄华占是感温型常规稻品种，2016—2017 年在铅山县汪二镇作中稻示范种植综合表现优良。结合示范情况，对黄华占的特征特性作简单总结，在此基础上提出高产栽培技术，以期为该品种的推广种植提供参考。
关键词 水稻；黄华占；特征特性；高产；栽培技术；江西铅山中图分类号 S511.048文献标识码 B文章编号 1007-5739（2018）13-0020-01
近年来，铅山县粮食生产紧紧围绕“稳产、优质、增效”的总体要求、大力实施优质稻推广，积极引导粮食生产由增产转向提质。我国杂交水稻技术世界领先、优质稻品种众多，在市场走势方面（尤其稻米行情清淡期），人们习惯性地北涨看长粒香、南涨看黄华占。黄华占是广东省农业科学院水稻研究所以黄新占/丰华占为亲本选育而成，分别通过粤、湘、鄂、浙、桂、琼等省审定。为了更好、更快地推广黄华占水稻，铅山县分别于2016 年、2017 年在汪二镇火田村试验示范种植黄华占近 5.87 hm^2 ，综合表现优良。现将黄华占水稻的特征特性及高产栽培技术介绍如下。
################
输出：
("entity"{tuple_delimiter}"黄华占"{tuple_delimiter}"work"{tuple_delimiter}"黄华占是一种感温型常规稻品种，由广东省农业科学院水稻研究所选育，通过多个省份审定，2016-2017年在铅山县汪二镇进行示范种植，表现优良。"){record_delimiter}
("entity"{tuple_delimiter}"2016—2017年"{tuple_delimiter}"date"{tuple_delimiter}"2016—2017年是黄华占在铅山县汪二镇进行示范种植的时间段。"){record_delimiter}
("entity"{tuple_delimiter}"铅山县"{tuple_delimiter}"location"{tuple_delimiter}"铅山县位于中国江西省，是黄华占水稻示范种植的地点之一。"){record_delimiter}
("entity"{tuple_delimiter}"汪二镇"{tuple_delimiter}"location"{tuple_delimiter}"汪二镇是铅山县的一个镇，2016-2017年在此进行了黄华占水稻的示范种植。"){record_delimiter}
("entity"{tuple_delimiter}"火田村"{tuple_delimiter}"location"{tuple_delimiter}"火田村是汪二镇的一个村庄，2016-2017年在此进行了黄华占水稻的试验示范种植。"){record_delimiter}
("entity"{tuple_delimiter}"广东省农业科学院水稻研究所"{tuple_delimiter}"organization"{tuple_delimiter}"广东省农业科学院水稻研究所是中国的一个科研机构，负责黄华占水稻的选育工作。"){record_delimiter}
("entity"{tuple_delimiter}"黄新占/丰华占"{tuple_delimiter}"work"{tuple_delimiter}"黄新占和丰华占是黄华占水稻的亲本，用于选育黄华占。"){record_delimiter}
("entity"{tuple_delimiter}"粤、湘、鄂、浙、桂、琼等省"{tuple_delimiter}"location"{tuple_delimiter}"这些省份通过了黄华占水稻的审定，表明该品种在这些地区具有良好的适应性和推广潜力。"){record_delimiter}
("entity"{tuple_delimiter}"高产栽培技术"{tuple_delimiter}"technology"{tuple_delimiter}"高产栽培技术是指为了提高黄华占水稻产量而采用的一系列农业技术措施。"){record_delimiter}
("entity"{tuple_delimiter}"稳产、优质、增效"{tuple_delimiter}"concept"{tuple_delimiter}"这是铅山县粮食生产的主要目标，强调了粮食生产的稳定、质量和效益。"){record_delimiter}
("entity"{tuple_delimiter}"优质稻推广"{tuple_delimiter}"mission"{tuple_delimiter}"优质稻推广是铅山县粮食生产的一个重要任务，旨在提高稻米的质量和市场竞争力。"){record_delimiter}
("entity"{tuple_delimiter}"杂交水稻技术"{tuple_delimiter}"technology"{tuple_delimiter}"杂交水稻技术是中国领先的世界级农业技术，用于提高水稻的产量和质量。"){record_delimiter}
("entity"{tuple_delimiter}"北涨看长粒香、南涨看黄华占"{tuple_delimiter}"concept"{tuple_delimiter}"这是市场对不同地区优质稻品种的习惯性关注点，北方面对长粒香，南方面对黄华占。"){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"2016—2017年"{tuple_delimiter}"黄华占在2016—2017年期间在铅山县进行了示范种植，展示了其优良的特性。"{tuple_delimiter}"时间, 植物生长"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"铅山县"{tuple_delimiter}"黄华占在铅山县进行了示范种植，表现出了优良的适应性和产量。"{tuple_delimiter}"地理位置, 植物适应性"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"汪二镇"{tuple_delimiter}"黄华占在汪二镇进行了示范种植，这是其在铅山县示范种植的一部分。"{tuple_delimiter}"地理位置, 植物适应性"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"火田村"{tuple_delimiter}"黄华占在火田村进行了试验示范种植，这是其在汪二镇示范种植的一部分。"{tuple_delimiter}"地理位置, 植物适应性"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"广东省农业科学院水稻研究所"{tuple_delimiter}"黄华占是由广东省农业科学院水稻研究所选育的，该研究所负责其研发工作。"{tuple_delimiter}"研发, 机构作用"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"黄新占/丰华占"{tuple_delimiter}"黄华占的亲本是黄新占和丰华占，这些亲本用于选育黄华占。"{tuple_delimiter}"遗传, 选育过程"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"粤、湘、鄂、浙、桂、琼等省"{tuple_delimiter}"黄华占通过了这些省份的审定，表明其在这些地区的适应性和推广潜力。"{tuple_delimiter}"地区适应性, 推广潜力"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"黄华占"{tuple_delimiter}"高产栽培技术"{tuple_delimiter}"高产栽培技术是为了提高黄华占水稻产量而开发的技术措施。"{tuple_delimiter}"技术应用, 产量提升"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"铅山县"{tuple_delimiter}"稳产、优质、增效"{tuple_delimiter}"铅山县的粮食生产目标是稳产、优质、增效，这些目标指导了黄华占的示范种植。"{tuple_delimiter}"生产目标, 地方政策"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"铅山县"{tuple_delimiter}"优质稻推广"{tuple_delimiter}"铅山县实施了优质稻推广计划，黄华占是该计划的一部分。"{tuple_delimiter}"政策实施, 品种推广"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"杂交水稻技术"{tuple_delimiter}"北涨看长粒香、南涨看黄华占"{tuple_delimiter}"杂交水稻技术的发展使得黄华占等优质稻品种在市场中受到关注。"{tuple_delimiter}"市场趋势, 技术影响"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"黄华占, 水稻种植, 高产栽培技术, 优质稻推广, 地区适应性, 市场趋势, 技术影响"){completion_delimiter}

-真实数据-
实体类型：{entity_types}
文本：{input_text}
################
输出：
"""

CONTINUE_EN = """MANY entities and relationships were missed in the last extraction.  Add them below using the same format:
"""

CONTINUE_ZH = """很多实体和关系在上一次的提取中可能被遗漏了。请在下面使用相同的格式添加它们："""

IF_LOOP_EN = """It appears some entities and relationships may have still been missed.  Answer YES | NO if there are still entities and relationships that need to be added.
"""

IF_LOOP_ZH = """看起来可能仍然遗漏了一些实体和关系。如果仍有实体和关系需要添加，请回答YES | NO。"""

KG_EXTRACTION_PROMPT = {
    "English": {
        "TEMPLATE": TEMPLATE_EN,
        "CONTINUE": CONTINUE_EN,
        "IF_LOOP": IF_LOOP_EN,
    },
    "Chinese": {
        "TEMPLATE": TEMPLATE_ZH,
        "CONTINUE": CONTINUE_ZH,
        "IF_LOOP": IF_LOOP_ZH,
    },
    "FORMAT": {
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>",
        "entity_types": "concept, date, location, keyword, organization, person, event, work, nature, artificial, science, technology, mission",
        "language": "English",
    },
}
