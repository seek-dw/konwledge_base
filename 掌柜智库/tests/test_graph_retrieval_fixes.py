from atguigu.query_process.nodes.node_graph_search import NodeGraphSearch
from atguigu.query_process.nodes.node_rrf import NodeRrf
from atguigu.tool.graph_store import GraphStore


class _Record:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _Session:
    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def run(self, query, **params):
        self.queries.append(query)
        if "[*1..6]" in query:
            return [_Record({"chunk_id": "chunk-6-hop", "path": ["A", "B", "C", "D", "E", "F"]})]
        return []


class _Driver:
    def __init__(self):
        self.session_instance = _Session()

    def session(self, **kwargs):
        return self.session_instance


class _GraphStore:
    def __init__(self):
        self.kwargs = None

    def find_related_chunks(self, entity_names, **kwargs):
        self.kwargs = kwargs
        return []


def test_graph_search_keeps_a_six_hop_path():
    driver = _Driver()

    hits = GraphStore(driver=driver).find_related_chunks(["A"], hops=6)

    assert hits == [{"chunk_id": "chunk-6-hop", "path": [["A", "B", "C", "D", "E", "F"]]}]
    assert any("[*1..6]" in query for query in driver.session_instance.queries)


def test_graph_store_defaults_to_three_hops():
    driver = _Driver()

    GraphStore(driver=driver).find_related_chunks(["A"])

    assert any("[*1..3]" in query for query in driver.session_instance.queries)


def test_graph_search_node_requests_six_hops_for_path_queries():
    graph_store = _GraphStore()

    NodeGraphSearch(graph_store=graph_store).process({
        "item_names": ["A"],
        "rewritten_query": "沿概念图给出机器学习到深度学习的主路径",
    })

    assert graph_store.kwargs["hops"] == 6


def test_graph_search_node_uses_three_hops_for_fact_queries():
    graph_store = _GraphStore()

    NodeGraphSearch(graph_store=graph_store).process({
        "item_names": ["A"],
        "rewritten_query": "A 是什么",
    })

    assert graph_store.kwargs["hops"] == 3


def test_rrf_deduplicates_graph_and_vector_results_by_chunk_id():
    result = NodeRrf().process({
        "embedding_chunks": [
            {"chunk_id": "chunk-A", "content": "same evidence", "source": "local"}
        ],
        "hyde_embedding_chunks": [],
        "graph_chunks": [
            {
                "id": 123,
                "chunk_id": "chunk-A",
                "content": "same evidence",
                "source": "graph",
                "graph_path": [["A", "B"]],
            }
        ],
    })["rrf_chunks"]

    assert len(result) == 1
    assert result[0]["chunk_id"] == "chunk-A"
    assert result[0]["graph_path"] == [["A", "B"]]
