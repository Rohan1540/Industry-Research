import os
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UseCaseGenerationAgent:
    def __init__(self, company_name):
        self.company_name = company_name
        self.research_agent = DynamicIndustryResearchAgent(company_name)
        self.research_agent.run()  # Run the initial research workflow
        self.report_data = self.research_agent.generate_report()  # Capture the research data

        # Initialize a text generator for generating relevant use cases
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            try:
                self.generator = pipeline("text-generation", model="gpt-3.5-turbo")
            except Exception as e:
                print(f"Error initializing generator: {e}")
                self.generator = None
        else:
            self.generator = None

    def analyze_industry_standards(self):
        """Analyze industry trends and standards based on initial research data."""
        industry_trends = self.report_data.get("AI Trends in Industry", "")
        industry_overview = self.report_data.get("Industry Overview", "")
        
        if self.generator and industry_trends and industry_overview:
            prompt = (
                f"In the {self.report_data['Industry']} industry, describe current AI, ML, and automation standards "
                f"and trends based on the following information:\n{industry_overview}\n{industry_trends}"
            )
            try:
                response = self.generator(prompt, max_length=150, num_return_sequences=1)
                standards_analysis = response[0]["generated_text"]
            except Exception as e:
                print(f"Error generating industry standards analysis: {e}")
                standards_analysis = "Unable to generate industry standards analysis."
        else:
            standards_analysis = "Insufficient data to analyze industry standards."

        return standards_analysis

    def generate_use_cases(self):
        """Generate AI and ML use cases for the company based on industry insights."""
        focus_areas = self.report_data.get("Company Focus Areas", "")
        industry = self.report_data.get("Industry", "")
        
        if self.generator and focus_areas and industry:
            prompt = (
                f"For a company in the {industry} sector with a focus on {focus_areas}, propose 3 relevant use cases "
                "leveraging GenAI, LLMs, and ML to improve operations, enhance customer satisfaction, and boost efficiency."
            )
            try:
                response = self.generator(prompt, max_length=200, num_return_sequences=1)
                use_case_proposals = response[0]["generated_text"]
            except Exception as e:
                print(f"Error generating use cases: {e}")
                use_case_proposals = "Unable to generate use case proposals."
        else:
            use_case_proposals = "Insufficient data to generate use case proposals."

        return use_case_proposals

    def generate_final_report(self):
        """Generate a final report with both industry standards analysis and use cases."""
        standards_analysis = self.analyze_industry_standards()
        use_cases = self.generate_use_cases()

        final_report = {
            "Industry Standards Analysis": standards_analysis,
            "Proposed AI/ML Use Cases": use_cases,
        }
        return final_report

    def save_final_report_to_csv(self, report, file_path):
        """Save the final report to a CSV file."""
        df = pd.DataFrame.from_dict(report, orient='index', columns=['Content'])
        df.to_csv(file_path, index=True, header=["Content"], encoding='utf-8')

    def run(self):
        """Run the use case generation workflow and save the final report."""
        print("Starting Use Case Generation...")
        final_report = self.generate_final_report()
        self.save_final_report_to_csv(final_report, "use_case_report.csv")
        print("Final report saved to use_case_report.csv")

# Usage example
if __name__ == "__main__":
    company_name = "Google"
    use_case_agent = UseCaseGenerationAgent(company_name=company_name)
    use_case_agent.run()
