import os
import requests
import openai
from bs4 import BeautifulSoup
from transformers import pipeline
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env
load_dotenv()
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

class DynamicIndustryResearchAgent:
    def __init__(self, company_name):
        self.company_name = company_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        # Initialize summarizer and classifier with error handling
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn") if self.api_key else None
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            print(f"Error initializing summarizer or classifier: {e}")
            self.summarizer = None
            self.classifier = None
        
        self.data = {
            "industry": None,
            "industry_overview": None,
            "company_focus_areas": None,
            "industry_ai_trends": "",
        }
        self.industry_candidates = ["Automotive", "Manufacturing", "Finance", "Retail", "Healthcare", "Technology"]

    def fetch_company_info(self):
        """Fetch brief information about the company to infer the industry."""
        search_query = f"{self.company_name} company overview"
        url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        headers = {"User-Agent": USER_AGENT}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            company_info = " ".join([p.text for p in soup.find_all("span")[:5]])
            return company_info
        except Exception as e:
            print(f"Error fetching company info: {e}")
            return None

    def infer_industry(self, company_info):
        """Infer the industry of the company based on the fetched information."""
        if company_info and self.classifier:
            classification = self.classifier(company_info, candidate_labels=self.industry_candidates)
            industry = classification['labels'][0]
            self.data["industry"] = industry
            print(f"Inferred Industry: {industry}")
        else:
            print("Unable to infer industry; company information missing or classifier unavailable.")

    def fetch_industry_info(self):
        """Fetch industry-specific news and insights using News API with fallback."""
        if self.data["industry"] and self.news_api_key:
            search_query = f"{self.data['industry']} industry trends"
            url = f"https://newsapi.org/v2/everything?q={search_query}&apiKey={self.news_api_key}"
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                news_data = response.json()
                
                if news_data["articles"]:
                    industry_overview = ". ".join([article["title"] for article in news_data["articles"][:5]])
                    self.data["industry_overview"] = industry_overview
                else:
                    self.data["industry_overview"] = "No specific overview found."
            except Exception as e:
                print(f"Error fetching industry info: {e}")
                self.data["industry_overview"] = "Unable to retrieve industry overview."

    def fetch_company_focus_areas(self):
        """Fetch the company's key offerings, strategies, and focus areas."""
        search_query = f"{self.company_name} strategic focus areas site:crunchbase.com"
        url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
        headers = {"User-Agent": USER_AGENT}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            focus_areas = " ".join([span.text for span in soup.find_all("span")[:10]])
            self.data["company_focus_areas"] = focus_areas if focus_areas else "Focus areas not found."
        except Exception as e:
            print(f"Error fetching company focus areas: {e}")
            self.data["company_focus_areas"] = "Unable to retrieve focus areas."

    def fetch_towards_data_science_trends(self):
        """Fetch AI trends from Towards Data Science."""
        url = "https://towardsdatascience.com/tagged/artificial-intelligence"
        headers = {"User-Agent": USER_AGENT}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("div", class_="postArticle-content")
            trends = "\n".join([post.find("h3").get_text() for post in articles[:5] if post.find("h3")])
            self.data["industry_ai_trends"] += f"\n\nTowards Data Science Insights: {trends}"
            print("Fetched AI trends from Towards Data Science.")
        except Exception as e:
            print(f"Error fetching AI trends from Towards Data Science: {e}")

    def analyze_trends(self):
        """Analyze and summarize trends in AI/GenAI applications within the industry."""
        if self.data["industry"]:
            self.fetch_towards_data_science_trends()
            if not self.data["industry_ai_trends"]:
                self.data["industry_ai_trends"] = "No trends data available."
        else:
            print("Industry not inferred; cannot analyze trends.")

    def summarize_content(self):
        """Use NLP summarization to create concise summaries of each category."""
        for key, content in self.data.items():
            if content and self.summarizer:
                if len(content.split()) > 10:
                    try:
                        max_len = min(len(content.split()) - 1, 50)
                        self.data[key] = self.summarizer(content, max_length=max_len, min_length=10, do_sample=False)[0]["summary_text"]
                    except Exception as e:
                        print(f"Error during summarization: {e}")
                else:
                    self.data[key] = content

    def generate_report(self):
        """Generate a structured report of the industry and company information."""
        report = {
            "Industry": self.data["industry"],
            "Industry Overview": self.data["industry_overview"],
            "Company Focus Areas": self.data["company_focus_areas"],
            "AI Trends in Industry": self.data["industry_ai_trends"],
        }
        return report

    def save_report_to_csv(self, report, file_path):
        """Save the report to a CSV file."""
        df = pd.DataFrame.from_dict(report, orient='index', columns=['Content'])
        df.to_csv(file_path, index=True, header=["Content"], encoding='utf-8')

    def run(self):
        """Run the complete research workflow."""
        print("Starting Dynamic Industry Research...")
        company_info = self.fetch_company_info()
        self.infer_industry(company_info)
        self.fetch_industry_info()
        self.fetch_company_focus_areas()
        self.analyze_trends()
        self.summarize_content()

        report = self.generate_report()
        self.save_report_to_csv(report, "industry_research_report.csv")
        print("Report saved to industry_research_report.csv")


class UseCaseGenerationAgent:
    def __init__(self, company_name):
        self.company_name = company_name
        self.research_agent = DynamicIndustryResearchAgent(company_name)
        self.research_agent.run()  # Run the initial research workflow
        self.report_data = self.research_agent.generate_report()  # Capture the research data
        self.generator = pipeline("text-generation", model="gpt2")  # You can choose any suitable model

    def analyze_industry_standards(self):
        """Analyze industry trends and standards based on initial research data."""
        industry_trends = self.report_data.get("AI Trends in Industry", "")
        industry_overview = self.report_data.get("Industry Overview", "")
        
        if industry_trends and industry_overview:
            prompt = (
                f"In the {self.report_data['Industry']} industry, describe current AI, ML, and automation standards "
                f"and trends based on the following information:\n{industry_overview}\n{industry_trends}\n\nAnalysis:"
            )
            try:
                response = self.generator(prompt, max_length=150, num_return_sequences=1, truncation=True)
                standards_analysis = response[0]['generated_text']
            except Exception as e:
                print(f"Error generating industry standards analysis: {e}")
                standards_analysis = "Unable to generate industry standards analysis."
        else:
            standards_analysis = "Insufficient data for industry standards analysis."

        return standards_analysis

    def generate_use_cases(self):
        """Generate use cases based on the standards analysis."""
        standards_analysis = self.analyze_industry_standards()

        # Define individual prompts for each use case to enhance clarity and specificity.
        prompts = [
            f"Based on the latest trends in the {self.report_data['Industry']} industry, generate an actionable use case for GenAI that optimizes customer support interactions.",
            f"Based on the latest trends in the {self.report_data['Industry']} industry, provide a use case for machine learning to improve operational efficiency in supply chain management.",
            f"Based on the latest trends in the {self.report_data['Industry']} industry, describe a use case where large language models can enhance personalized marketing for customers."
        ]

        use_cases = []
        for prompt in prompts:
            try:
                response = self.generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
                use_case = response[0]['generated_text'].strip()
                use_cases.append(use_case)
            except Exception as e:
                print(f"Error generating use case for prompt '{prompt}': {e}")
                use_cases.append("Unable to generate use case.")

        return use_cases

    def save_use_cases_to_csv(self, use_cases, file_path="use_cases.csv"):
        """Save the generated use cases to a CSV file."""
        df = pd.DataFrame(use_cases, columns=["Generated Use Cases"])
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Use cases saved to {file_path}")

    def run(self):
        """Execute the use case generation workflow and save results."""
        print("Generating use cases...")
        use_cases = self.generate_use_cases()
        self.save_use_cases_to_csv(use_cases)



class ResourceAssetCollectionAgent:
    def __init__(self, use_cases):
        self.use_cases = use_cases

    def search_datasets(self, use_case):
        """Generate dataset search URLs for each use case."""
        keywords = "+".join(use_case.split()[:5])  # Use the first few words as search keywords

        # Dataset search URLs for Kaggle, Hugging Face, and GitHub
        kaggle_url = f"https://www.kaggle.com/search?q={keywords}"
        huggingface_url = f"https://huggingface.co/models?search={keywords}"
        github_url = f"https://github.com/search?q={keywords}+dataset"

        return {
            "Kaggle": kaggle_url,
            "HuggingFace": huggingface_url,
            "GitHub": github_url
        }

    def collect_resources(self):
        """Collect resources for each use case."""
        resources = {}
        for use_case in self.use_cases:
            dataset_links = self.search_datasets(use_case)
            resources[use_case] = dataset_links
        return resources

    def propose_genai_solutions(self):
        """Propose GenAI solutions based on the context of each use case."""
        genai_solutions = []
        for use_case in self.use_cases:
            solution = ""
            if "customer support" in use_case.lower():
                solution = (
                    "- **AI-Powered Chat Systems**: Implement a chatbot using large language models to assist in customer "
                    "support, offering instant responses and query resolutions.\n"
                    "- **Document Search**: Use document retrieval models to allow support agents to quickly access relevant "
                    "documentation or past support cases."
                )
            elif "supply chain" in use_case.lower():
                solution = (
                    "- **Automated Report Generation**: Use AI to generate reports on supply chain metrics, trends, and "
                    "predictive insights, optimizing decision-making.\n"
                    "- **Predictive Analysis**: Use machine learning models to forecast supply chain demands and mitigate risks."
                )
            elif "personalized marketing" in use_case.lower():
                solution = (
                    "- **Customer Segmentation**: Use GenAI models to segment customers based on preferences and behaviors.\n"
                    "- **Content Generation**: Use AI to generate personalized marketing content across various channels, "
                    "enhancing engagement and conversion."
                )
            if solution:
                genai_solutions.append(f"### For Use Case: {use_case}\n{solution}\n")
        
        return genai_solutions

    def save_resources_to_markdown(self, resources, genai_solutions, file_path="resources.md"):
        """Save the collected resources and GenAI solution proposals to a markdown file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Resource Collection and GenAI Solutions\n\n")

            # Add resource links
            f.write("## Dataset Resources\n")
            for use_case, links in resources.items():
                f.write(f"### Use Case: {use_case}\n")
                for platform, url in links.items():
                    f.write(f"- [{platform}]({url})\n")
                f.write("\n")

            # Add GenAI solution proposals
            f.write("## Proposed GenAI Solutions\n")
            for solution in genai_solutions:
                f.write(solution + "\n")

        print(f"Resources and GenAI solutions saved to {file_path}")

    def run(self):
        """Execute the resource collection and proposal workflow."""
        print("Collecting dataset resources and proposing GenAI solutions...")
        
        # Step 1: Collect dataset resources for each use case
        resources = self.collect_resources()
        
        # Step 2: Generate GenAI solution proposals
        genai_solutions = self.propose_genai_solutions()
        
        # Step 3: Save to markdown file
        self.save_resources_to_markdown(resources, genai_solutions)




class FinalProposalAgent:
    def __init__(self, company_name, use_cases):
        self.company_name = company_name
        self.use_cases = use_cases

    def search_datasets(self, use_case):
        """Generate dataset search URLs for each use case."""
        keywords = "+".join(use_case.split()[:5])  # Use the first few words as search keywords

        # Dataset search URLs for Kaggle, Hugging Face, and GitHub
        kaggle_url = f"https://www.kaggle.com/search?q={keywords}"
        huggingface_url = f"https://huggingface.co/models?search={keywords}"
        github_url = f"https://github.com/search?q={keywords}+dataset"

        return {
            "Kaggle": kaggle_url,
            "HuggingFace": huggingface_url,
            "GitHub": github_url
        }

    def recommend_additional_resources(self):
        """Provide links to industry-specific reports and articles."""
        resources = {
            "McKinsey": "https://www.mckinsey.com/industries/technology/our-insights",
            "Deloitte": "https://www2.deloitte.com/global/en/insights/industry.html",
            "Nexocode": "https://www.nexocode.com/resources",
            "General Industry Trends": [
                "https://www.forbes.com/sites/bernardmarr/2023/01/10/top-5-ai-trends-in-2023/",
                "https://www.gartner.com/en/insights/artificial-intelligence",
            ]
        }
        return resources

    def collect_top_use_cases(self):
        """Collect top use cases based on relevance to company goals."""
        # Simulate selecting the top use cases relevant to the company's goals
        top_use_cases = {
            "Customer Support Optimization": {
                "Description": (
                    "Implement an AI-powered chatbot to assist with customer support interactions, "
                    "improving response time and enhancing customer satisfaction."
                ),
                "Reference": "https://www.forbes.com/sites/forbestechcouncil/2023/02/10/ai-driven-customer-support-trends/"
            },
            "Supply Chain Efficiency": {
                "Description": (
                    "Use machine learning for predictive analysis in supply chain management, "
                    "reducing operational costs and increasing efficiency in inventory management."
                ),
                "Reference": "https://www.mckinsey.com/industries/supply-chain"
            },
            "Personalized Marketing": {
                "Description": (
                    "Utilize large language models to enhance customer segmentation and personalize marketing campaigns, "
                    "leading to improved engagement and conversion rates."
                ),
                "Reference": "https://www2.deloitte.com/global/en/insights/topics/marketing-and-sales.html"
            }
        }
        return top_use_cases

    def save_proposal_to_markdown(self, top_use_cases, dataset_resources, additional_resources, file_path="final_proposal.md"):
        """Save the final proposal, including use cases, resources, and additional references, to a markdown file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Final Proposal for AI and ML Use Cases\n\n")
            f.write("This document outlines the top use cases for AI and ML applications, relevant datasets, and additional resources.\n\n")

            # Top Use Cases
            f.write("## Top Use Cases\n")
            for use_case, details in top_use_cases.items():
                f.write(f"### {use_case}\n")
                f.write(f"- **Description**: {details['Description']}\n")
                f.write(f"- **Reference**: [{details['Reference']}]({details['Reference']})\n\n")

            # Dataset Resources
            f.write("## Dataset Resources\n")
            for use_case, links in dataset_resources.items():
                f.write(f"### Use Case: {use_case}\n")
                for platform, url in links.items():
                    f.write(f"- [{platform}]({url})\n")
                f.write("\n")

            # Additional Resources
            f.write("## Additional Resources\n")
            for source, link in additional_resources.items():
                if isinstance(link, list):  # Handle multiple links for some resources
                    f.write(f"### {source}\n")
                    for url in link:
                        f.write(f"- [{url}]({url})\n")
                else:
                    f.write(f"- **{source}**: [{link}]({link})\n")
            f.write("\n")

        print(f"Final proposal saved to {file_path}")

    def run(self):
        """Execute the final proposal workflow."""
        print("Compiling final proposal with top use cases, dataset resources, and additional references...")

        # Step 1: Collect top use cases based on company goals
        top_use_cases = self.collect_top_use_cases()
        
        # Step 2: Search for datasets relevant to each top use case
        dataset_resources = {use_case: self.search_datasets(use_case) for use_case in top_use_cases.keys()}
        
        # Step 3: Gather additional resources from industry-specific reports
        additional_resources = self.recommend_additional_resources()
        
        # Step 4: Save the proposal to a markdown file
        self.save_proposal_to_markdown(top_use_cases, dataset_resources, additional_resources)


# Full execution example
if __name__ == "__main__":
    company_name = "IBM"
    use_case_generator = UseCaseGenerationAgent(company_name)
    use_case_generator.run()
    
    # Use cases generated from the previous agent
    generated_use_cases = use_case_generator.generate_use_cases()
    
    # Initialize the resource collection agent with generated use cases
    resource_agent = ResourceAssetCollectionAgent(generated_use_cases)
    resource_agent.run()


    final_proposal_agent = FinalProposalAgent(company_name, generated_use_cases)
    final_proposal_agent.run()