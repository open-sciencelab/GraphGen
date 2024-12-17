from models import WikiSearch, OpenAIModel
from models.storage.base_storage import BaseGraphStorage
from templates import SEARCH_JUDGEMENT_PROMPT
from utils import logger

async def search_wikipedia(llm_client: OpenAIModel, wiki_search_client: WikiSearch, knowledge_graph_instance: BaseGraphStorage,) -> dict:
    """
    Search wikipedia for entities

    :param llm_client: LLM model
    :param wiki_search_client: wiki search client
    :param knowledge_graph_instance: knowledge graph instance
    :return: nodes with search results
    """
    nodes = await knowledge_graph_instance.get_all_nodes()
    wiki_data = {}
    for node in nodes:
        entity_name = node[0].strip('"')
        description = node[1]["description"]
        search_results = await wiki_search_client.search(entity_name)
        if not search_results:
            continue
        examples = "\n".join(SEARCH_JUDGEMENT_PROMPT["EXAMPLES"])
        search_results.append("None of the above")

        search_results_str = "\n".join([f"{i + 1}. {sr}" for i, sr in enumerate(search_results)])
        prompt = SEARCH_JUDGEMENT_PROMPT["TEMPLATE"].format(
            examples=examples,
            entity_name=entity_name,
            description=description,
            search_results=search_results_str,
        )
        response = llm_client.generate_answer(prompt)
        response = response.strip()

        try:
            response = int(response)
            if response < 1 or response > len(search_results):
                response = None
        except ValueError:
            response = None
        if response is None or response == len(search_results):
            continue

        search_result = search_results[response - 1]

        summary = await wiki_search_client.summary(search_result)
        if summary:
            logger.info(f"Wiki search result for {entity_name}: {summary}")
            wiki_data[entity_name] = summary
    return wiki_data
