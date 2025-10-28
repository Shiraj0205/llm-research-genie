import os
import sys
from datetime import datetime
from typing import List, Optional
from exception.custom_exception import ResearchAnalystException
from langgraph.types import Send

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage,HumanMessage , SystemMessage, get_buffer_string
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

from docx import Document
from logger import GLOBAL_LOGGER
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Adjust the import path to include the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)  

from research_and_analyst.backend_server.models import (
    Analyst,
    Perspectives,
    GenerateAnalystsState,
    InterviewState,
    ResearchGraphState,
    SearchQuery,
    )

from research_and_analyst.utils.model_loader import ModelLoader
from dotenv import load_dotenv
from research_and_analyst.prompt_lib.prompts import *

def build_interview_graph(llm, tavily_search=None):
    
    memory = MemorySaver()

    def generation_question(state: InterviewState):
        analyst = state["analyst"]
        messages = state["messages"]

        try:
            system_prompt = ANALYST_ASK_QUESTIONS.render(goals=analyst.persona)
            question = llm.invoke([SystemMessage(content=system_prompt)] + messages)
            return {"messages": [question]}
        except Exception as e:
            raise ResearchAnalystException("Failed to generate analyst question", e)
        

    def search_web(state: InterviewState):
        try:
            structure_llm = llm.with_structured_output(SearchQuery)
            search_query = structure_llm.invoke([GENERATE_SEARCH_QUERY] + state["messages"])

            search_docs = tavily_search.invoke(search_query.search_query)

            if not search_docs:
                return {"context": ["[No search results found.]"]}

            formatted = "\n\n---\n\n".join(
                [
                    f'<Document href="{doc.get("url", "#")}"/>\n{doc.get("content", "")}\n</Document>'
                    for doc in search_docs
                ]
            )
            return {"context": [formatted]}

        except Exception as e:
            raise ResearchAnalystException("Failed during web search execution", e)


    def generate_answer(state: InterviewState):
        analyst = state["analyst"]
        messages = state["messages"]
        context = state.get("context", ["[No context available.]"])

        try:
            system_prompt = GENERATE_ANSWERS.render(goals=analyst.persona, context=context)
            answer = llm.invoke([SystemMessage(content=system_prompt)] + messages)
            answer.name = "expert"
            return {"messages": [answer]}

        except Exception as e:
            raise ResearchAnalystException("Failed to generate expert answer", e)


    def save_interview(state: InterviewState):
        try:
            messages = state["messages"]
            interview = get_buffer_string(messages)
            return {"interview": interview}

        except Exception as e:
            raise ResearchAnalystException("Failed to save interview transcript", e)


    def write_section(state: InterviewState):
        context = state.get("context", ["[No context available.]"])
        analyst = state["analyst"]

        try:
            system_prompt = WRITE_SECTION.render(focus=analyst.description)
            section = llm.invoke(
                [SystemMessage(content=system_prompt)]
                + [HumanMessage(content=f"Use this source to write your section: {context}")]
            )
            return {"sections": [section.content]}

        except Exception as e:
            raise ResearchAnalystException("Failed to generate report section", e)

    interview_builder = StateGraph(InterviewState)

    interview_builder.add_node("ask_question",generation_question)
    interview_builder.add_node("search_web",search_web)
    interview_builder.add_node("generate_answer",generate_answer)
    interview_builder.add_node("save_interview",save_interview)
    interview_builder.add_node("write_section",write_section)

    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question","search_web")
    interview_builder.add_edge("search_web","generate_answer")
    interview_builder.add_edge("generate_answer","save_interview")
    interview_builder.add_edge("save_interview","write_section")
    interview_builder.add_edge("write_section",END)

    return interview_builder.build(checkpointer=memory)


class ReportGenerator:
    def __init__(self, llm: ModelLoader):
        load_dotenv()
        self.llm = llm
        self.memory = MemorySaver()
        self.tavily_search = TavilySearchResults()
        self.logger = GLOBAL_LOGGER.bind(module="ReportGenerator")

    
    def create_analyst(self, state: GenerateAnalystsState):
        """
        Create a set of AI analyst personas based on the research topic and any provided feedback.
        """
        topic = state["topic"]
        max_analysts = state["max_analysts"]
        human_analyst_feedback = state.get("human_analyst_feedback", "")

        try:
            self.logger.info("Creating analyst personas", topic=topic)
            structured_llm = self.llm.with_structured_output(Perspectives)
            system_prompt = CREATE_ANALYSTS_PROMPT.render(
                topic=topic, max_analysts=max_analysts,
                human_analyst_feedback=human_analyst_feedback,
            )
            analysts = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate the set of analysts."),
            ])
            self.logger.info("Analysts created", count=len(analysts.analysts))
            return {"analysts": analysts.analysts}
        except Exception as e:
            self.logger.error("Error creating analysts", error=str(e))
            raise ResearchAnalystException("Failed to create analysts", e)


    def human_feedback(self):
        pass


    def generate_report(self):
        pass


    def write_report(self, state: ResearchGraphState):
        """
        Write the research report based on the gathered information.
        """
        sections = state.get("sections", [])
        topic = state.get("topic", "")

        if not sections or len(sections) == 0:
            sections = ["No sections were created during the interviews."]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        system_prompt = REPORT_WRITER_INSTRUCTIONS.render(topic=topic)
        report = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="\n\n".join(sections))
            ])
        return {"content": report.content}


    def write_introduction(self, state: ResearchGraphState):
        """
        Write the introduction for the research report.
        """
        sections = state.get("sections", [])
        topic = state.get("topic", "")

        formatted_str_sections = "\n\n".join([f"{s}" for s in sections])

        system_prompt = INTRO_CONCLUSION_INSTRUCTIONS.render(
                topic=topic, formatted_str_sections=formatted_str_sections
                )
        
        intro = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Write the report introduction")
        ])

        return {"introduction": intro.content}

    def write_conclusion(self):
        pass

    def finalize_report(self):
        pass

    def save_report(self, final_report: str, topic: str, format: str = "docs", save_dir: Optional[str] = None):
        pass

    def _save_as_pdf(self, text: str, file_path: str):
        pass

    def _save_as_docx(self, text: str, file_path: str):
        pass

    def build_research_graph(self):

        builder = StateGraph(ResearchGraphState)
        interview_graph = build_interview_graph(self.llm, self.tavily_search)

        def initiate_all_interviews(state: ResearchGraphState):
            topic = state.get("topic", "Untitled Topic")
            analysts = state.get("analysts", [])
            if not analysts:
                self.logger.warning("No analysts found — skipping interviews")
                return END
            return [
                Send(
                    "conduct_interview",
                    {
                        "analyst": analyst,
                        "messages": [HumanMessage(content=f"So, let's discuss about {topic}.")],
                        "max_num_turns": 2,
                        "context": [],
                        "interview": "",
                        "sections": [],
                    },
                )
                for analyst in analysts
            ]

        builder.add_node("create_analysts", self.create_analyst)
        builder.add_node("human_feedback", self.human_feedback)
        builder.add_node("initiate_all_interviews", interview_graph)
        builder.add_node("write_report", self.write_report)
        builder.add_node("write_introduction", self.write_introduction)
        builder.add_node("write_conclusion", self.write_conclusion)
        builder.add_node("finalize_report", self.finalize_report)

        # Logic
        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["conduct_interview"])
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
        builder.add_edge("finalize_report", END)

        return builder.build()


if __name__ == "__main__":
    model_loader = ModelLoader()
    llm = model_loader.load_llm()
    llm.invoke("hello")

    report_generator = ReportGenerator(llm)
    graph = report_generator.build_research_graph()

    topic = "The impact of artificial intelligence on modern healthcare."
    thread = {"configurable" : {"thread_id": "1"}}

    for _ in graph.stream({"topic": topic, "max_analysts": 3}, thread, stream_mode="values"):
            pass
    
    state = graph.get_state(thread)
    feedback = input("\n Enter your feedback or press Enter to continue: ").strip()
    graph.update_state(thread, {"human_analyst_feedback": feedback}, as_node="human_feedback")

    for _ in graph.stream(None, thread, stream_mode="values"):
        pass

    final_state = graph.get_state(thread)
    final_report = final_state.values.get("final_report")

    if final_report:
        report_generator.logger.info("Report generated successfully")
        report_generator.save_report(final_report, topic, "docx")
        report_generator.save_report(final_report, topic, "pdf")
    else:
        report_generator.logger.error("No report generated.")