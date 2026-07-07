import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from workflow import AgentWorkflow


def main():
    print("Hello from biomedical-ner!")
    workflow = AgentWorkflow()
    workflow.save_graph_diagram("graph.png")


if __name__ == "__main__":
    main()
