import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

class Visualizer:
    """
    A class to capture and visualize causal relationships in articles using Neo4j and Pyvis.
    This class requires a Neo4j database to be running and accessible.
    Capure causal relationships using a pre-trained model from Hugging Face.
    The model used is "Babelscape/rebel-large" for extracting relationships.
    """

    def __init__(self, file_path, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
        """
        :param file_path: Path to the file containing articles.
        """
        self.file_path = file_path
        self.articles = []
        self.entity = set()
        self.relation = set()

        # Filter for causal relationships (selected from the availble relations in the model)
        causal_keywords = [
            "encodes", "use", "uses", "based on", "connects with", "endemic to", "influenced by",
            "followed by", "follows", "has cause", "has effect", "inception",
        ]
        self.causal_keywords = set(causal_keywords)  # for faster lookup

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Load the model and tokenizer for catching relationships
        model_name = "Babelscape/rebel-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()

    def read_json(self):
        """Read articles from the JSON file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.articles = json.load(f)
        except FileNotFoundError:
            print(f"File {self.file_path} not found.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file {self.file_path}.")
        return self.articles
    
    def process(self):
        """Process all articles in the JSON file. Store the results in Neo4j."""
        self.read_json()
        for article in self.articles:
            if article['content']:
                content = article['content']
                # Extract causal relationships
                causal_relationships = self.extract_causal_relationships(content)

                for relationship in causal_relationships:
                    entity_one = relationship['head']
                    entity_two = relationship['tail']
                    relation = relationship['type']
                    self.entity.add(entity_one)
                    self.entity.add(entity_two)
                    self.relation.add(relation)

                    # Store in Neo4j
                    with self.driver.session() as session:
                        session.run("MERGE (a:Entity {name: $entity_one}) "
                                    "MERGE (b:Entity {name: $entity_two}) "
                                    "MERGE (a)-[:RELATION {type: $relation}]->(b)",
                                    entity_one=entity_one, entity_two=entity_two, relation=relation)
        return
    
    
    def extract_causal_relationships(self, text):
        """
        Extract causal relationships from the text using the model.
        :param text: The text to analyze.
        :return: List of (subject, relation, object) tuples representing causal relationships.
        """
        try:
            # Process text in chunks of ~500 words to avoid truncation issues
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            all_triples = []
            gen_kwargs = {
                "max_length": 256,
                "length_penalty": 0,
                "num_beams": 3,
                "num_return_sequences": 3,
            }
            
            for chunk in chunks:
                # Process each chunk
                # Tokenizer text
                model_inputs = self.tokenizer(chunk, max_length=256, padding=True, truncation=True, return_tensors = 'pt')

                # Generate
                generated_tokens = self.model.generate(
                    **model_inputs,
                    **gen_kwargs,
                )

                # Extract text
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

                # Extract triplets
                for sentence in decoded_preds:
                    triplets = self.extract_triplets(sentence)
                    all_triples.extend(triplets)
            
            
            return [t for t in all_triples if any(k in t['type'].lower() for k in self.causal_keywords)]
        
        except Exception as e:
            print(f"Error extracting relationships: {e}")
            return []

    def extract_triplets(self, text):
        # Extract triplets from the text.
        # output: [{'head': subject, 'type': relation, 'tail': object_}]
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
        return triplets

    def visualize(self, output_file="graph.html"):
        """Generate a visualization of the knowledge graph
        :param output_file: Path to save the HTML visualization
        """
        # Query all relationships from Neo4j
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity)-[r:RELATION]->(e2:Entity)
                RETURN e1.name AS source, r.type AS relation, e2.name AS target
            """)

            print("Extracted relationships:")
            print(result)
            
            # Create a graph
            G = nx.DiGraph()
            
            # Add edges with relation as label
            for record in result:
                G.add_edge(record["source"], record["target"], 
                        title=record["relation"], label=record["relation"])
            
            # Create interactive visualization
            net = Network(notebook=False, height="750px", width="100%", 
                        bgcolor="#222222", font_color="white")
            net.from_nx(G)
            net.save_graph(output_file)
            
            print(f"Visualization saved to {output_file}")
            
if __name__ == "__main__":
    # Initialize the Visualizer with the path to the JSON file
    tool = Visualizer("data/extracted_articles.json")

    # process the json file and store the results in Neo4j
    tool.process()

    # fetch the data from Neo4j and visualize it
    tool.visualize("knowledge_graph.html")

    # Close the Neo4j driver
    tool.close()