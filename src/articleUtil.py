from optimUtil import DspyField, create_output_model_class
import pandas as pd
from tqdm import tqdm
from loguru import logger
import subprocess
import tempfile
import re
from pylatexenc.latex2text import LatexNodes2Text
import numpy as np
from typing import Literal
import json
import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

MARKER_WORKS_NUM=1

class TeXProcessor:
    def __init__(self, main_tex_path):
        self.main_tex_path = main_tex_path
        self.tex_dir = os.path.dirname(main_tex_path)
        self.merged_content = ""
        self.parsed_sections = {}
        self.tables = []

    def read_tex_file(self, file_path):
        """Read TeX file content"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read file: {file_path}, error: {e}")
            return ""

    def merge_input_files(self):
        """Merge all files introduced via input in the main file"""
        main_content = self.read_tex_file(self.main_tex_path)
        content = main_content

        input_pattern = r"\\input\{([^}]*)\}"
        input_matches = re.findall(input_pattern, content)

        for input_file in input_matches:
            input_path = os.path.join(self.tex_dir, input_file)
            if not input_path.endswith(".tex"):
                input_path += ".tex"

            if os.path.exists(input_path):
                input_content = self.read_tex_file(input_path)
                content = content.replace(f"\\input{{{input_file}}}", input_content)

        self.merged_content = content
        return content

    def parse_content(self):
        """Parse merged TeX content, parsing sections in order from title downward"""
        if not self.merged_content:
            self.merge_input_files()

        self.parse_tables()

        for table in self.tables:
            self.merged_content = self.merged_content.replace(table["content"], "")

        title_pattern = r"\\title\{{([^}]*)\}}"
        title_match = re.search(title_pattern, self.merged_content)
        if title_match:
            self.parsed_sections["title"] = title_match.group(1).replace("\\", "")

        abstract_pattern = r"\\begin\{{abstract\}}(.*?)\\end\{{abstract\}}"
        abstract_match = re.search(abstract_pattern, self.merged_content, re.DOTALL)
        if abstract_match:
            abstract_content = abstract_match.group(1).strip()
            self.parsed_sections["abstract"] = LatexNodes2Text().latex_to_text(
                abstract_content
            )

        keyword_pattern = r"\\keywords\{{([^}]*)\}}"
        keyword_match = re.search(keyword_pattern, self.merged_content)
        if keyword_match:
            self.parsed_sections["keywords"] = keyword_match.group(1).split(",")
            self.parsed_sections["keywords"] = [
                kw.strip() for kw in self.parsed_sections["keywords"]
            ]

        section_pattern = r"\\section\*?{(.*?)}"
        section_matches = list(re.finditer(section_pattern, self.merged_content))

        for i, match in enumerate(section_matches):
            section_title = match.group(1).strip()
            section_key = section_title.lower().replace(" ", "_")

            start_pos = match.end()

            if i < len(section_matches) - 1:
                end_pos = section_matches[i + 1].start()
            else:
                end_pattern = r"(?:\\bibliography|\\end\{{document\}})"
                end_match = re.search(end_pattern, self.merged_content[start_pos:])
                if end_match:
                    end_pos = start_pos + end_match.start()
                else:
                    end_pos = len(self.merged_content)

            section_content = self.merged_content[start_pos:end_pos].strip()
            self.parsed_sections[section_key] = LatexNodes2Text().latex_to_text(
                section_content
            )

        return self.parsed_sections

    def parse_tables(self):
        """Parse tables and preserve original LaTeX code"""
        tables = []
        start_pos = 0
        table_count = 0

        while True:
            table_start_pattern = r"\\begin\{[^\}]*table[^\}]*\}"
            match = re.search(table_start_pattern, self.merged_content[start_pos:])
            begin_table_pos = start_pos + match.start() if match else -1
            if begin_table_pos != -1:
                table_end_pattern = r"\\end\{[^\}]*table[^\}]*\}"
                end_match = re.search(
                    table_end_pattern, self.merged_content[begin_table_pos:]
                )
                end_table_pos = begin_table_pos + end_match.start() if end_match else -1
                if end_table_pos == -1:
                    break
                end_table_pos += len("\\end{table}")
                table_content = self.merged_content[begin_table_pos:end_table_pos]
            else:
                tabular_start_pattern = r"\\begin\{[^\}]*tabular[^\}]*\}"
                match = re.search(
                    tabular_start_pattern, self.merged_content[start_pos:]
                )
                begin_table_pos = start_pos + match.start() if match else -1
                if begin_table_pos == -1:
                    break
                tabular_end_pattern = r"\\end\{[^\}]*tabular[^\}]*\}"
                end_match = re.search(
                    tabular_end_pattern, self.merged_content[begin_table_pos:]
                )
                end_table_pos = begin_table_pos + end_match.start() if end_match else -1
                if end_table_pos == -1:
                    break
                end_table_pos += len("\\end{tabular}")
                table_content = self.merged_content[begin_table_pos:end_table_pos]

            caption = f"Table {table_count + 1}"
            caption_pos = self.merged_content.rfind("\\caption{{", 0, begin_table_pos)
            if caption_pos != -1:
                caption_end_pos = self.merged_content.find("}}", caption_pos)
                if caption_end_pos != -1:
                    caption = self.merged_content[
                        caption_pos + len("\\caption{{") : caption_end_pos
                    ]
                    caption = caption.replace("\\", "")

            tables.append({"caption": caption, "content": table_content})
            table_count += 1
            start_pos = end_table_pos

        self.tables = tables

    def convert_to_markdown(self):
        """Convert parsed content to markdown"""
        if not self.parsed_sections:
            self.parse_content()

        md_content = ""

        if "title" in self.parsed_sections:
            md_content += f"# {self.parsed_sections['title']}\n\n"

        if "abstract" in self.parsed_sections:
            md_content += f"## Abstract\n{self.parsed_sections['abstract']}\n\n"

        if "keywords" in self.parsed_sections:
            md_content += (
                f"## Keywords\n{', '.join(self.parsed_sections['keywords'])}\n\n"
            )

        for section, content in self.parsed_sections.items():
            if section not in ["title", "abstract", "keywords"]:
                md_content += f"## {section.capitalize()}\n{content}\n\n"

        if self.tables:
            md_content += "## Tables\n"
            for i, table in enumerate(self.tables):
                md_content += f"### Table {i + 1}: {table['caption']}\n\n```latex\n{table['content']}\n```\n\n"

        return md_content

    def process(self):
        """Execute complete processing workflow"""
        self.merge_input_files()
        self.parse_content()
        return self.convert_to_markdown()


class ParsedMarkdown:
    def __init__(self, file_path):
        self.file_path = file_path
        self.content = self._read_file()
        self.sections = self._parse_sections()
        # self.figures = self._parse_figures()
        # self.tables = self._parse_tables()
        
    # Pattern map for section identification
    PATTERN_MAP = {
        "introduction": "Introduction",
        "background": "Introduction",
        "experimental procedure": "Method",
        "method": "Method",
        "result": "Result",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
        "funding": "Funding",
        "acknowledgment": "Acknowledgement",
        "reference": "Reference",
        "conflict of interest": "Conflict of Interest",
        "supplementary material": "Supporting Information",
        "conflicts of interest": "Conflict of Interest",
        "acknowledgement": "Acknowledgement",
        "supporting information": "Supporting Information",
        "supplementary information": "Supporting Information",
        "abbreviations": "Abbreviations",
    }

    def _read_file(self):
        with open(self.file_path, "r", encoding="utf-8") as file:
            return file.readlines()

    def _parse_md_structure(self):
        primary_structure = {}
        max_header_words = 6
        for index, line in enumerate(self.content):
            if line is not None and line.startswith("#"):
                for pattern, section in self.PATTERN_MAP.items():
                    if (
                        pattern in line.lower()
                        and len(line.strip().split(" ")) <= max_header_words
                    ):
                        if section not in primary_structure:
                            primary_structure[section] = []
                        primary_structure[section].append((line.strip(), index))
                        break
        # in some cases, the article does not have a section header for the introduction
        structure = {"Introduction": 0}
        for section, lines in primary_structure.items():
            if len(lines) == 1:
                structure[section] = lines[0][1]
                continue
            else:
                line_lengths = [len(line.strip().split(" ")) for line, _ in lines]
                min_length = min(line_lengths)
                indexs = [i for i, x in enumerate(line_lengths) if x == min_length]
                structure[section] = lines[indexs[-1]][1]
        return structure

    def _parse_sections(self):
        structure = self._parse_md_structure()
        break_point = {value: key for key, value in structure.items()}
        section = {"Others": ""}
        current_section = "Others"
        # reference_pattern  = r'\[\d+\]'
        temp_lines = []
        for index, line in enumerate(self.content):
            if index in break_point:
                current_section = break_point[index]
                section[current_section] = ""
            # deal with reference
            if current_section == "Reference":
                temp_lines.append(line)
            else:
                section[current_section] += line
        last_line = ""
        for line in reversed(temp_lines):
            if line.startswith("- "):
                last_line = line
                break
        section["Reference"] = ""
        if "Result" not in section:
            section["Result"] = ""
        if last_line != "":
            section["Reference"] += "".join(
                temp_lines[0 : temp_lines.index(last_line) + 1]
            )
            section["Result"] += "".join(temp_lines[temp_lines.index(last_line) + 1 :])
        else:
            section["Reference"] += "".join(temp_lines)
            
        # Check for missing sections from PATTERN_MAP and add them with empty string values
        for pattern, section_name in self.PATTERN_MAP.items():
            if section_name not in section:
                section[section_name] = ""
                
        return section


class ScienceDirectXmlParser:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.namespaces = {
            "ce": "http://www.elsevier.com/xml/common/dtd",
            "xocs": "http://www.elsevier.com/xml/xocs/dtd",
            "ja": "http://www.elsevier.com/xml/ja/dtd",
            "dc": "http://purl.org/dc/elements/1.1/",
            "prism": "http://prismstandard.org/namespaces/basic/2.0/",
            "dcterms": "http://purl.org/dc/terms/",
            "cals": "http://www.elsevier.com/xml/common/cals/dtd",
        }
        self.content = {}
        self.all_tables = self.extract_all_tables()

    def extract_title(self):
        title = self.root.find(".//dc:title", self.namespaces)
        if title is not None:
            return title.text
        title = self.root.find(".//ce:title", self.namespaces)
        if title is not None:
            return title.text
        return "Untitled Article"

    def extract_abstract(self):
        abstract = self.root.find(".//ce:abstract", self.namespaces)
        if abstract is not None:
            return "".join(abstract.itertext())
        return "No abstract available"

    def extract_keywords(self):
        keywords = []
        kw_elements = self.root.findall(".//ce:keyword", self.namespaces)
        if kw_elements:
            for kw in kw_elements:
                text_elem = kw.find("./ce:text", self.namespaces)
                if text_elem is not None:
                    keywords.append(" ".join(text_elem.itertext()))
        else:
            kw_elements = self.root.findall(".//dcterms:subject", self.namespaces)
            for kw in kw_elements:
                keywords.append(" ".join(kw.itertext()))
        return keywords

    def extract_all_tables(self):
        """Extract all tables from the XML and store them by their ID."""
        tables = {}
        table_elements = self.root.findall(".//ce:table", self.namespaces)
        for table in table_elements:
            table_id = table.get("id")
            if table_id:
                tables[table_id] = self._convert_table_to_html(table)
        return tables

    def extract_sections(self):
        sections = {}
        top_level_sections = self.root.findall(
            ".//ce:sections/ce:section", self.namespaces
        )

        for section in top_level_sections:
            self._process_section(section, sections, [])

        return sections

    def _process_section(self, section_element, sections, parent_titles):
        label = section_element.find("./ce:label", self.namespaces)
        title_element = section_element.find("./ce:section-title", self.namespaces)

        if title_element is None:
            title = "Unnamed Section"
        else:
            title = title_element.text
            if title is not None and "introduction" in title.lower():
                title = "Introduction"
            else:
                title = ""

        if label is not None and label.text:
            full_title = f"{label.text}. {title}"
        else:
            full_title = title

        current_titles = parent_titles + [full_title]
        section_key = " > ".join(current_titles)

        content = []
        tables = []
        for elem in section_element:
            if elem.tag.endswith("section"):
                continue

            if elem.tag.endswith("para"):
                para_text = "".join([elemt.strip() for elemt in elem.itertext()])

                cross_refs = elem.findall(".//ce:cross-ref", self.namespaces)
                if cross_refs:
                    for ref in cross_refs:
                        refid = ref.get("refid")
                        if refid and refid in self.all_tables:
                            table_html = self.all_tables[refid]
                            if not any(table_id == refid for table_id, _ in tables):
                                tables.append((refid, table_html))
                            ref_text = "".join(ref.itertext())
                            para_text = para_text.replace(ref_text, f"[Table {refid}]")

                content.append(para_text + "\n\n")
            elif elem.tag.endswith("section-title") and elem != title_element:
                sub_title = elem.text
                content.append(f"\n## {sub_title}\n")
            elif elem.tag.endswith("table"):
                table_id = elem.get("id")
                table_html = self._convert_table_to_html(elem)
                if not any(table_id == t[0] for t in tables):
                    tables.append((table_id, table_html))
            elif elem.tag.endswith("figure"):
                figure_label = elem.find("./ce:label", self.namespaces)
                figure_caption = elem.find("./ce:caption", self.namespaces)
                if figure_label and figure_caption:
                    figure_html = f"<figure>\n<figcaption>{figure_label.text}: {''.join(figure_caption.itertext())}</figcaption>\n</figure>"
                    content.append(f"{figure_html}")

        for _, table_html in tables:
            content.append(f"{table_html}")
            content.append("\n\n")

        sections[section_key] = " ".join(content)

        sub_sections = section_element.findall("./ce:section", self.namespaces)
        for sub_section in sub_sections:
            self._process_section(sub_section, sections, current_titles)

    def _convert_table_to_html(self, table_element):
        soup = BeautifulSoup("<table></table>", "html.parser")
        table = soup.table

        caption = table_element.find("./ce:caption", self.namespaces)
        if caption:
            caption_tag = soup.new_tag("caption")
            caption_tag.string = " ".join(caption.itertext())
            table.append(caption_tag)

        tgroup = table_element.find(
            "{http://www.elsevier.com/xml/common/cals/dtd}tgroup"
        )
        print(f"tgroup:{tgroup}")
        if not tgroup:
            return self._convert_old_table_to_html(table_element, soup, table)

        colspecs = tgroup.findall(
            "{http://www.elsevier.com/xml/common/cals/dtd}colspec", self.namespaces
        )
        col_props = {}
        for idx, colspec in enumerate(colspecs):
            colname = colspec.get("colname")
            col_props[idx] = {
                "align": colspec.get("align"),
                "width": colspec.get("width"),
            }
            if colname:
                col_props[colname] = col_props[idx]

        thead_element = tgroup.find(
            "{http://www.elsevier.com/xml/common/cals/dtd}thead", self.namespaces
        )
        if thead_element:
            thead = soup.new_tag("thead")
            header_rows = thead_element.findall(
                "{http://www.elsevier.com/xml/common/cals/dtd}row", self.namespaces
            )
            for row in header_rows:
                tr = soup.new_tag("tr")
                entries = row.findall(
                    "{http://www.elsevier.com/xml/common/dtd}entry", self.namespaces
                )
                for i, entry in enumerate(entries):
                    props = col_props.get(i, {})

                    th = soup.new_tag("th")
                    if props.get("align"):
                        th["align"] = props["align"]
                    if props.get("width"):
                        th["width"] = props["width"]
                    if entry.get("align"):
                        th["align"] = entry.get("align")
                    if entry.get("valign"):
                        th["valign"] = entry.get("valign")

                    th.string = " ".join(entry.itertext())
                    tr.append(th)
                thead.append(tr)
            table.append(thead)

        tbody_element = tgroup.find(
            "{http://www.elsevier.com/xml/common/cals/dtd}tbody", self.namespaces
        )
        if tbody_element:
            tbody = soup.new_tag("tbody")
            body_rows = tbody_element.findall(
                "{http://www.elsevier.com/xml/common/cals/dtd}row", self.namespaces
            )
            for row in body_rows:
                tr = soup.new_tag("tr")
                entries = row.findall(
                    "{http://www.elsevier.com/xml/common/dtd}entry", self.namespaces
                )
                for i, entry in enumerate(entries):
                    props = col_props.get(i, {})

                    td = soup.new_tag("td")
                    if props.get("align"):
                        td["align"] = props["align"]
                    if props.get("width"):
                        td["width"] = props["width"]
                    if entry.get("align"):
                        td["align"] = entry.get("align")
                    if entry.get("valign"):
                        td["valign"] = entry.get("valign")
                    if entry.get("role") == "rowhead":
                        td["role"] = "rowhead"

                    td.string = " ".join(entry.itertext())
                    tr.append(td)
                tbody.append(tr)
            table.append(tbody)

        return str(table)

    def _convert_old_table_to_html(self, table_element, soup, table):
        rows = table_element.findall("./ce:tr", self.namespaces)
        if rows:
            thead = soup.new_tag("thead")
            tr = soup.new_tag("tr")
            headers = rows[0].findall("./ce:th", self.namespaces)
            for header in headers:
                th = soup.new_tag("th")
                th.string = " ".join(header.itertext())
                tr.append(th)
            thead.append(tr)
            table.append(thead)

            tbody = soup.new_tag("tbody")
            for row in rows[1:]:
                tr = soup.new_tag("tr")
                cells = row.findall("./ce:td", self.namespaces)
                for cell in cells:
                    td = soup.new_tag("td")
                    td.string = " ".join(cell.itertext())
                    tr.append(td)
                tbody.append(tr)
            table.append(tbody)

        return str(table)

    def extract_acknowledgment(self):
        acknowledgment = self.root.find(".//ce:acknowledgment", self.namespaces)
        if acknowledgment is not None:
            return " ".join(acknowledgment.itertext())
        return "No acknowledgment available"

    def extract_references(self):
        references = []
        bib_element = self.root.find(".//ce:bibliography", self.namespaces)
        if bib_element is not None:
            ref_elements = bib_element.findall(".//ce:reference", self.namespaces)
            for i, ref in enumerate(ref_elements, 1):
                ref_text = " ".join(ref.itertext())
                references.append(f"[{i}] {ref_text}")
        return "\n".join(references)

    def parse(self):
        """Parse the XML file and extract all relevant content"""
        self.content["title"] = self.extract_title()
        self.content["abstract"] = self.extract_abstract()
        self.content["keywords"] = self.extract_keywords()
        self.content["sections"] = self.extract_sections()
        self.content["acknowledgment"] = self.extract_acknowledgment()
        self.content["references"] = self.extract_references()
        return self.content

    def to_markdown(self, output_path):
        """Convert parsed content to markdown and save to file"""
        if not self.content:
            self.parse()

        with open(output_path, "w", encoding="utf-8") as f:
            # Write title
            f.write(f"# {self.content['title']}\n\n")

            # Write abstract
            f.write("## Abstract\n")
            f.write(f"{self.content['abstract']}\n\n")

            # Write keywords
            if self.content["keywords"]:
                f.write("## Keywords\n")
                f.write(", ".join(self.content["keywords"]) + "\n\n")

            # Write sections
            for title, content in self.content["sections"].items():
                level = title.count(">")
                prefix = "#" * (level + 2)
                title = title.split(">")[-1].strip()
                f.write(f"{prefix} {title}\n")
                f.write(f"{content}\n\n")

            # Write acknowledgment
            f.write("## Acknowledgment\n")
            f.write(f"{self.content['acknowledgment']}\n\n")

            # Write references
            f.write("## References\n")
            f.write(f"{self.content['references']}\n")


class PubMedCentralXmlParser:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.namespaces = {
            "ali": "http://www.niso.org/schemas/ali/1.0/",
            "mml": "http://www.w3.org/1998/Math/MathML",
            "xlink": "http://www.w3.org/1999/xlink",
        }
        self.content = {}
        self.all_tables = self.extract_all_tables()
        self.added_table_ids = set()

    def extract_title(self):
        title = self.root.find(".//article-title")
        if title is not None:
            return "".join(title.itertext())
        return "Untitled Article"

    def extract_abstract(self):
        abstract = self.root.find(".//abstract")
        if abstract is not None:
            return "".join(abstract.itertext())
        return "No abstract available"

    def extract_keywords(self):
        keywords = []
        kw_group = self.root.find(".//kwd-group")
        if kw_group is not None:
            kw_elements = kw_group.findall(".//kwd")
            for kw in kw_elements:
                keywords.append("".join(kw.itertext()))
        return keywords

    def extract_all_tables(self):
        """Extract all tables from the XML and store them by their ID."""
        tables = {}
        table_elements = self.root.findall(".//table-wrap")
        for table in table_elements:
            table_id = table.get("id")
            if table_id:
                tables[table_id] = self._convert_table_to_html(table)
        print(tables)
        return tables

    def extract_sections(self):
        sections = {}
        body = self.root.find(".//body")
        if body is not None:
            top_level_sections = body.findall("./sec")
            for section in top_level_sections:
                self._process_section(section, sections, [])
        return sections

    def _process_paragraph_content(self, paragraph_element):
        """parse paragraph content"""
        content_parts = []

        full_text = "".join(paragraph_element.itertext()).strip()

        if paragraph_element:
            for child in paragraph_element:
                if child.tag == "table-wrap":
                    table_id = child.get("id")
                    if not table_id or table_id not in self.added_table_ids:
                        table_html = self._convert_table_to_html(child)
                        content_parts.append(table_html)
                        if table_id:
                            self.added_table_ids.add(table_id)
                elif child.tag == "fig":
                    figure_label = child.find("./label")
                    figure_caption = child.find("./caption")
                    if figure_label and figure_caption:
                        figure_html = f"<figure>\n<figcaption>{figure_label.text}: {''.join(figure_caption.itertext())}</figcaption>\n</figure>"
                        content_parts.append(figure_html)
                elif child.tag == "xref" and child.get("ref-type") == "table":
                    ref_id = child.get("rid")
                    if ref_id and ref_id not in self.added_table_ids:
                        if hasattr(self, "all_tables") and ref_id in self.all_tables:
                            content_parts.append(self.all_tables[ref_id])
                            self.added_table_ids.add(ref_id)

        if full_text:
            content_parts.append(full_text)

        return "\n".join(content_parts)

    def _process_section(self, section_element, sections, parent_titles):
        title_element = section_element.find("./title")
        if title_element is None:
            # For the top-level body element
            current_titles = parent_titles
            section_key = ""
        else:
            title = title_element.text
            if title is not None and "introduction" in title.lower():
                title = "Introduction"
            elif title is None:
                title = ""
            current_titles = parent_titles + [title]
            section_key = " > ".join(current_titles)

        content = []
        tables = []
        for elem in section_element:
            # if elem.tag == 'sec':
            #     self._process_section(elem, sections, current_titles)
            #     continue

            if elem.tag == "p":
                para_content = self._process_paragraph_content(elem)
                content.append(para_content + "\n\n")
            elif elem.tag == "title" and elem != title_element:
                sub_title = elem.text
                content.append(f"\n## {sub_title}\n")
            elif elem.tag == "table-wrap":
                table_id = elem.get("id")
                if not table_id or table_id not in self.added_table_ids:
                    table_html = self._convert_table_to_html(elem)
                    if table_id:
                        self.added_table_ids.add(table_id)
                    tables.append((table_id, table_html))
            elif elem.tag == "fig":
                figure_label = elem.find("./label")
                figure_caption = elem.find("./caption")
                if figure_label and figure_caption:
                    figure_html = f"<figure>\n<figcaption>{figure_label.text}: {''.join(figure_caption.itertext())}</figcaption>\n</figure>"
                    content.append(f"{figure_html}")

        for _, table_html in tables:
            content.append(f"{table_html}")
            content.append("\n\n")

        if section_key not in sections:
            sections[section_key] = ""
        sections[section_key] += " ".join(content)

        sub_sections = section_element.findall("./sec")

        for sub_section in sub_sections:
            self._process_section(sub_section, sections, current_titles)

    def _convert_table_to_html(self, table_element):
        soup = BeautifulSoup("<table></table>", "html.parser")
        table = soup.table

        caption = table_element.find("./caption")
        if caption:
            caption_tag = soup.new_tag("caption")
            caption_tag.string = "".join(caption.itertext())
            table.append(caption_tag)

        table_content = table_element.find(".//table")
        if not table_content:
            return str(table)

        # Process header rows
        thead = soup.new_tag("thead")
        header_rows = table_content.findall("./thead/tr")
        if not header_rows:
            # Check if first row is header
            all_rows = table_content.findall("./tr")
            if all_rows:
                header_rows = [all_rows[0]]
                body_rows = all_rows[1:]
            else:
                header_rows = []
                body_rows = []
        else:
            body_rows = table_content.findall("./tbody/tr")

        for row in header_rows:
            tr = soup.new_tag("tr")
            th_cells = row.findall("./th")
            td_cells = row.findall("./td")
            cells = th_cells if th_cells else td_cells
            for cell in cells:
                th = soup.new_tag("th")
                if cell.get("align"):
                    th["align"] = cell.get("align")
                if cell.get("valign"):
                    th["valign"] = cell.get("valign")
                th.string = "".join(cell.itertext())
                tr.append(th)
            thead.append(tr)
        if thead.find_all():  # Only add thead if it has content
            table.append(thead)

        # Process body rows
        tbody = soup.new_tag("tbody")
        for row in body_rows:
            tr = soup.new_tag("tr")
            cells = row.findall("./td")
            for cell in cells:
                td = soup.new_tag("td")
                if cell.get("align"):
                    td["align"] = cell.get("align")
                if cell.get("valign"):
                    td["valign"] = cell.get("valign")
                td.string = "".join(cell.itertext())
                tr.append(td)
            tbody.append(tr)
        if tbody.find_all():  # Only add tbody if it has content
            table.append(tbody)

        return str(table)

    def extract_acknowledgment(self):
        acknowledgment = self.root.find(".//ack")
        if acknowledgment is not None:
            return "".join(acknowledgment.itertext())
        return "No acknowledgment available"

    def extract_references(self):
        references = []
        ref_list = self.root.find(".//ref-list")
        if ref_list is not None:
            ref_elements = ref_list.findall(".//ref")
            for i, ref in enumerate(ref_elements, 1):
                ref_text = "".join(ref.itertext())
                references.append(f"[{i}] {ref_text}")
        return "\n".join(references)

    def parse(self):
        """Parse the XML file and extract all relevant content"""
        self.content["title"] = self.extract_title()
        self.content["abstract"] = self.extract_abstract()
        self.content["keywords"] = self.extract_keywords()
        self.content["sections"] = self.extract_sections()
        self.content["acknowledgment"] = self.extract_acknowledgment()
        self.content["references"] = self.extract_references()
        return self.content

    def to_markdown(self, output_path):
        """Convert parsed content to markdown and save to file"""
        if not self.content:
            self.parse()

        with open(output_path, "w", encoding="utf-8") as f:
            # Write title
            f.write(f"# {self.content['title']}\n\n")

            # Write abstract
            f.write("## Abstract\n")
            f.write(f"{self.content['abstract']}\n\n")

            # Write keywords
            if self.content["keywords"]:
                f.write("## Keywords\n")
                f.write(", ".join(self.content["keywords"]) + "\n\n")

            # Write sections
            for title, content in self.content["sections"].items():
                level = title.count("> ")
                prefix = "#" * (level + 2)
                title = title.split("> ")[-1].strip()
                f.write(f"{prefix} {title}\n")
                f.write(f"{content}\n\n")

            # Write acknowledgment
            if self.content["acknowledgment"] != "No acknowledgment available":
                f.write("## Acknowledgment\n")
                f.write(f"{self.content['acknowledgment']}\n\n")

            # Write references
            if self.content["references"]:
                f.write("## References\n")
                f.write(f"{self.content['references']}\n")


def split_md(folder_path, save_path):
    """
    Split markdown files in folder structure into JSON files with parsed sections.

    Args:
        folder_path (str): Path to folder containing markdown files (either in subfolders or directly in folder)
        save_path (str): Path to save the generated JSON files
    """
    folder_path = os.path.abspath(os.path.normpath(folder_path))
    save_path = os.path.abspath(os.path.normpath(save_path))

    # Validate input paths
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    elif not os.path.isdir(save_path):
        raise ValueError(f"Save path is not a directory: {save_path}")

    # Get all markdown files to process
    markdown_files = []

    # Check for subdirectories first
    subdirs = [
        d
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    if subdirs:
        # Process markdown files in subdirectories (original behavior)
        for subdir in subdirs:
            md_filename = f"{subdir}.md"
            md_path = os.path.join(folder_path, subdir, md_filename)
            if os.path.exists(md_path):
                markdown_files.append((subdir, md_path))
    else:
        # No subdirectories, process markdown files directly in folder
        md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
        for md_file in md_files:
            file_name = os.path.splitext(md_file)[0]
            md_path = os.path.join(folder_path, md_file)
            markdown_files.append((file_name, md_path))

    if not markdown_files:
        logger.warning(f"No markdown files found in {folder_path}")
        return

    processed_count = 0
    error_count = 0

    for file_name, md_path in tqdm(markdown_files, desc="Processing markdown files"):
        try:
            # Parse markdown file
            md_obj = ParsedMarkdown(md_path)

            # Construct output JSON file path
            json_filename = f"{file_name}.json"
            json_path = os.path.join(save_path, json_filename)

            # Save parsed sections to JSON file with proper encoding
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(md_obj.sections, f)

            processed_count += 1

        except FileNotFoundError as e:
            error_count += 1
            logger.error(f"File not found error processing {file_name}: {e}")
        except Exception as e:
            error_count += 1
            logger.error(f"Unexpected error processing {file_name}: {e}")

    logger.info(
        f"Processing complete. Successfully processed: {processed_count}, Errors: {error_count}"
    )

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")


def parse_article_to_md(folder_path, save_path):
    """
    Convert PDF articles in the specified folder to Markdown format and save them.

    Args:
        folder_path (str): Source folder path containing PDF files
        save_path (str): Target folder path for saving converted Markdown files

    Returns:
        list: Paths to the generated markdown files

    Raises:
        ValueError: If the provided folder paths are invalid
    """
    folder_path = os.path.abspath(os.path.normpath(folder_path))
    save_path = os.path.abspath(os.path.normpath(save_path))
    config = {"output_format": "markdown", "disable_multiprocessing": True}

    # Validate input paths
    if not os.path.exists(folder_path):
        raise ValueError(f"Source folder does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Source path is not a directory: {folder_path}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    elif not os.path.isdir(save_path):
        raise ValueError(f"Save path is not a directory: {save_path}")

    def _run_marker_cli_for_folder(in_folder: str, out_folder: str, cfg: dict):
        if MARKER_WORKS_NUM == "AUTO":
            args = [
            "marker",
            in_folder,
            "--output_dir",
            out_folder,
            "--output_format",
            cfg.get("output_format", "markdown"),
        ]
        else:
            args = [
                "marker",
                in_folder,
                "--output_dir",
                out_folder,
                "--workers",
                str(MARKER_WORKS_NUM),
                "--output_format",
                cfg.get("output_format", "markdown"),
            ]
        if cfg.get("disable_multiprocessing", False):
            args.append("--disable_multiprocessing")
        tmp_cfg_path = None
        if cfg:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tmp_cfg_path = tmp_file.name
            tmp_file.close()
            with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                import json as _json
                _json.dump(cfg, f)
            args.extend(["--config_json", tmp_cfg_path])
        try:
            _workers_val = None if str(MARKER_WORKS_NUM).strip().upper() == "AUTO" else int(MARKER_WORKS_NUM)
        except Exception:
            _workers_val = None
        if _workers_val == 1:
            logger.info(
                "To ensure smooth execution, the worker count is set to 1. You can change MARKER_WORKS_NUM in src/articleUtil.py to increase concurrent PDF processing. If MARKER_WORKS_NUM is 'AUTO', the MARKER tool will set workers automatically."
            )
        logger.info(f"Running marker convert tool with {MARKER_WORKS_NUM} workers. Worker initialization may take time, please wait patiently.")
        try:
            completed = subprocess.run(args, stdout=None, stderr=None, text=True, check=True)
            if completed.stdout:
                logger.debug(completed.stdout)
            if completed.stderr:
                logger.debug(completed.stderr)
        finally:
            if tmp_cfg_path and os.path.exists(tmp_cfg_path):
                try:
                    os.unlink(tmp_cfg_path)
                except Exception:
                    pass

    _run_marker_cli_for_folder(folder_path, save_path, config)
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return []

    converted_files = []
    error_count = 0

    for file in pdf_files:
        file_name = file.split(".")[0]
        if not os.path.exists(os.path.join(save_path, file_name)):
            os.mkdir(os.path.join(save_path, file_name))
        try:
            output_path = os.path.join(save_path, file_name, file_name + ".md")
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Converted file not found: {output_path}")
            converted_files.append(output_path)
            logger.info(f"Successfully converted {file} to markdown")
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing {file}: {e}")
            continue

    logger.info(
        f"Processing complete. Successfully converted: {len(converted_files)}, Errors: {error_count}"
    )

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")

    return converted_files


def parse_pubmed_xml_to_md(folder_path, save_path):
    """
    Parse all PubMed Central XML files in a folder and convert them to markdown format.

    Args:
        folder_path (str): Path to the folder containing XML files
        save_path (str): Path to save the converted markdown files

    Returns:
        list: Paths to the generated markdown files
    """
    # Validate input paths
    folder_path = os.path.abspath(os.path.normpath(folder_path))
    save_path = os.path.abspath(os.path.normpath(save_path))

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    elif not os.path.isdir(save_path):
        raise ValueError(f"Save path is not a directory: {save_path}")

    # Get all XML files in the folder
    xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]

    if not xml_files:
        logger.warning(f"No XML files found in {folder_path}")
        return []

    converted_files = []
    error_count = 0

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        try:
            # Construct full XML file path
            xml_path = os.path.join(folder_path, xml_file)

            # Determine output path
            file_name = os.path.splitext(xml_file)[0]
            output_path = os.path.join(save_path, f"{file_name}.md")

            # Parse XML and convert to markdown
            parser = PubMedCentralXmlParser(xml_path)
            parser.to_markdown(output_path)
            converted_files.append(output_path)
            logger.info(f"Successfully converted {xml_path} to markdown")

        except Exception as e:
            error_count += 1
            logger.error(f"Error converting {xml_path} to markdown: {e}")
            continue

    logger.info(
        f"Processing complete. Successfully converted: {len(converted_files)}, Errors: {error_count}"
    )

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")

    return converted_files


def parse_tex_to_md_batch(folder_path, save_path):
    """
    Parse all TeX files in a folder and convert them to markdown format.

    Args:
        folder_path (str): Path to the folder containing TeX files
        save_path (str): Path to save the converted markdown files

    Returns:
        list: Paths to the generated markdown files
    """
    # Validate input paths
    folder_path = os.path.abspath(os.path.normpath(folder_path))
    save_path = os.path.abspath(os.path.normpath(save_path))

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    elif not os.path.isdir(save_path):
        raise ValueError(f"Save path is not a directory: {save_path}")

    # Get all TeX files in the folder
    tex_files = []
    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)):
            for file in os.listdir(os.path.join(folder_path, folder)):
                if file.endswith(".tex"):
                    tex_files.append(os.path.join(folder_path, folder, file))

    if not tex_files:
        logger.warning(f"No TeX files found in {folder_path}")
        return []

    converted_files = []
    error_count = 0

    for tex_path in tqdm(tex_files, desc="Processing TeX files"):
        try:
            tex_file = os.path.basename(tex_path)
            # Determine output path
            file_name = os.path.splitext(tex_file)[0]
            output_path = os.path.join(save_path, f"{file_name}.md")

            # Process TeX file and convert to markdown
            processor = TeXProcessor(tex_path)
            md_content = processor.process()

            # Save markdown content
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            converted_files.append(output_path)
            logger.info(f"Successfully converted {tex_path} to markdown")

        except Exception as e:
            error_count += 1
            logger.error(f"Error converting {tex_path} to markdown: {e}")
            continue

    logger.info(
        f"Processing complete. Successfully converted: {len(converted_files)}, Errors: {error_count}"
    )

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")

    return converted_files


def parse_xml_to_md(folder_path, save_path):
    """
    Parse all ScienceDirect XML files in a folder and convert them to markdown format.

    Args:
        folder_path (str): Path to the folder containing XML files
        save_path (str): Path to save the converted markdown files

    Returns:
        list: Paths to the generated markdown files
    """
    # Validate input paths
    folder_path = os.path.abspath(os.path.normpath(folder_path))
    save_path = os.path.abspath(os.path.normpath(save_path))

    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    elif not os.path.isdir(save_path):
        raise ValueError(f"Save path is not a directory: {save_path}")

    # Get all XML files in the folder
    xml_files = [f for f in os.listdir(folder_path) if f.endswith(".xml")]

    if not xml_files:
        logger.warning(f"No XML files found in {folder_path}")
        return []

    converted_files = []
    error_count = 0

    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        try:
            # Construct full XML file path
            xml_path = os.path.join(folder_path, xml_file)

            # Determine output path
            file_name = os.path.splitext(xml_file)[0]
            output_path = os.path.join(save_path, f"{file_name}.md")

            # Parse XML and convert to markdown
            parser = ScienceDirectXmlParser(xml_path)
            parser.to_markdown(output_path)
            converted_files.append(output_path)
            logger.info(f"Successfully converted {xml_path} to markdown")

        except Exception as e:
            error_count += 1
            logger.error(f"Error converting {xml_path} to markdown: {e}")
            continue

    logger.info(
        f"Processing complete. Successfully converted: {len(converted_files)}, Errors: {error_count}"
    )

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")

    return converted_files


def convert_md_to_json(folder_path, save_path):
    """
    Convert markdown files in folder structure into JSON files with raw content.

    Args:
        folder_path (str): Path to folder containing markdown files
                       Can contain either subfolders with markdown files
                       or markdown files directly in the folder
        save_path (str): Path to save the generated JSON files
    """
    folder_path = os.path.abspath(os.path.normpath(folder_path))
    save_path = os.path.abspath(os.path.normpath(save_path))

    # Validate input paths
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    elif not os.path.isdir(save_path):
        raise ValueError(f"Save path is not a directory: {save_path}")

    # Get all subdirectories in the folder
    subdirs = [
        d
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    # Collect markdown files to process
    markdown_files = []

    if subdirs:
        # Process subdirectories (traditional mode)
        for subdir in subdirs:
            md_filename = f"{subdir}.md"
            md_path = os.path.join(folder_path, subdir, md_filename)
            if os.path.exists(md_path):
                markdown_files.append((subdir, md_path))
    else:
        # No subdirectories, process markdown files directly in folder
        md_files = [f for f in os.listdir(folder_path) if f.endswith(".md")]
        for md_file in md_files:
            file_name = os.path.splitext(md_file)[0]
            md_path = os.path.join(folder_path, md_file)
            markdown_files.append((file_name, md_path))

    if not markdown_files:
        logger.warning(f"No markdown files found in {folder_path}")
        return

    processed_count = 0
    error_count = 0

    for file_name, md_path in tqdm(markdown_files, desc="Processing markdown files"):
        try:
            # Read markdown file content
            with open(md_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            # Construct output JSON file path
            json_filename = f"{file_name}.json"
            json_path = os.path.join(save_path, json_filename)

            # Save raw content to JSON file
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"Document": markdown_content}, f)

            processed_count += 1
            logger.debug(f"Successfully processed: {file_name}")

        except FileNotFoundError as e:
            error_count += 1
            logger.error(f"File not found error processing {file_name}: {e}")
        except Exception as e:
            error_count += 1
            logger.error(f"Unexpected error processing {file_name}: {e}")

    logger.info(
        f"Processing complete. Successfully processed: {processed_count}, Errors: {error_count}"
    )

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors during processing")


def build_optm_set_from_article_and_extracted_information(
    json_path: str,
    dataset: str,
    article_parts: list[
        Literal["Title","Abstract","Introduction", "Method", "Result", "Discussion", "Conclusion"]
    ],
    fields: list[DspyField],
    multiple: bool,
    article_field: str,
):
    """
    to build a optm dataset for prompt optimization.
    :param json_path: article json folders
    :param dataset: curated dataset path
    :param article_parts: parts of article to extract
    :param fields: fields to extract
    :param multiple: whether to extract multiple extract objects
    """
    # file_path_check
    if not os.path.exists(json_path):
        raise ValueError("Invalid json folder path")
    if not os.path.exists(dataset):
        raise FileNotFoundError("Invalid dataset path")

    # read extracted dataset
    try:
        if dataset.endswith(".json"):
            df = pd.read_json(dataset)
        elif dataset.endswith(".csv"):
            df = pd.read_csv(dataset)
        elif dataset.endswith(".tsv"):
            df = pd.read_csv(dataset, sep="\t")
        elif dataset.endswith(".xlsx"):
            df = pd.read_excel(dataset)
        else:
            logger.error(f"Error reading file {dataset}: Invalid dataset format")
            raise ValueError("Invalid dataset format")
    except Exception as e:
        logger.error(f"Error reading file {dataset}: {e}")
        raise
    # replace "" to None
    df = df.replace("", None)
    df.replace({np.nan: None}, inplace=True)
    result_dict = dict()
    if multiple:
        output_class = create_output_model_class(fields)
    else:
        pass
    for _, row in tqdm(df.iterrows(), total=len(df)):
        article_index = row[article_field]
        if f"{article_index}.json" in os.listdir(json_path):
            if article_index not in result_dict:
                json_path_article = os.path.join(json_path, f"{article_index}.json")
                result_dict[article_index] = {"article_field": article_index}
                with open(json_path_article, "r") as f:
                    current_file = json.load(f)
                    for article_part in article_parts:
                        result_dict[article_index][article_part] = (
                            current_file[article_part]
                            if article_part in current_file
                            else ""
                        )
                if multiple:
                    result_dict[article_index]["extracted_information"] = []
            if multiple:
                result_dict[article_index]["extracted_information"].append(
                    output_class(**{item.name: row[item.name] for item in fields})
                )
            else:
                for item in fields:
                    result_dict[article_index][item.name] = row[item.name]
    return list(result_dict.values())

def build_optm_set_from_document_and_extracted_information(
    json_path: str,
    dataset: str,
    fields: list[DspyField],
    multiple: bool,
    article_field: str,
):
    """
    to build a dataset for prompt optimization.
    :param json_path: document json folders
    :param dataset: curated dataset path
    :param fields: fields to extract
    :param multiple: whether to extract multiple extract objects
    """
    # file_path_check
    if not os.path.exists(json_path):
        raise ValueError("Invalid json folder path")
    if not os.path.exists(dataset):
        raise FileNotFoundError("Invalid dataset path")

    # read extracted dataset
    try:
        if dataset.endswith(".json"):
            df = pd.read_json(dataset)
        elif dataset.endswith(".csv"):
            df = pd.read_csv(dataset)
        elif dataset.endswith(".tsv"):
            df = pd.read_csv(dataset, sep="\t")
        elif dataset.endswith(".xlsx"):
            df = pd.read_excel(dataset)
        else:
            logger.error(f"Error reading file {dataset}: Invalid dataset format")
            raise ValueError("Invalid dataset format")
    except Exception as e:
        logger.error(f"Error reading file {dataset}: {e}")
        raise
    # replace "" to None
    df = df.replace("", None)
    df.replace({np.nan: None}, inplace=True)
    result_dict = dict()
    if multiple:
        output_class = create_output_model_class(fields)
    else:
        pass
    for _, row in tqdm(df.iterrows(), total=len(df)):
        article_index = row[article_field]
        if f"{article_index}.json" in os.listdir(json_path):
            if article_index not in result_dict:
                json_path_article = os.path.join(json_path, f"{article_index}.json")
                result_dict[article_index] = {"article_field": article_index}
                with open(json_path_article, "r") as f:
                    current_file = json.load(f)
                    result_dict[article_index]["Document"] = (
                        current_file["Document"] if "Document" in current_file else ""
                    )
                if multiple:
                    result_dict[article_index]["extracted_information"] = []
            if multiple:
                result_dict[article_index]["extracted_information"].append(
                    output_class(**{item.name: row[item.name] for item in fields})
                )
            else:
                for item in fields:
                    result_dict[article_index][item.name] = row[item.name]
    return list(result_dict.values())

def merge_json_files_to_dataset(json_folder_path, output_path="_dataset.json"):
    """
    Merge all JSON files in the specified directory into a single dataset.
    Each file's content becomes a record with an added 'article_field' key
    containing the filename (without extension), and the result is saved to _dataset.json.
    
    Args:
        json_folder_path (str): Path to the directory containing JSON files
        output_path (str): Path for the output dataset file, defaults to "_dataset.json"
    
    Returns:
        str: Full path to the output file
    
    Raises:
        ValueError: If the specified path is invalid
    """
    # Validate input path
    json_folder_path = os.path.abspath(os.path.normpath(json_folder_path))
    
    if not os.path.exists(json_folder_path):
        raise ValueError(f"Folder path does not exist: {json_folder_path}")
    if not os.path.isdir(json_folder_path):
        raise ValueError(f"Path is not a directory: {json_folder_path}")
    
    # Get all JSON files
    json_files = [f for f in os.listdir(json_folder_path) if f.endswith(".json")]
    
    if not json_files:
        logger.warning(f"No JSON files found in directory {json_folder_path}")
        return ""
    
    # Merge JSON files
    dataset = []
    error_count = 0
    
    for json_file in tqdm(json_files, desc="Merging JSON files"):
        try:
            # Get filename without extension
            file_name = os.path.splitext(json_file)[0]
            
            # Read JSON file content
            json_path = os.path.join(json_folder_path, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                file_content = json.load(f)
            
            # Create record with added article_field key
            record = file_content.copy()
            record["article_field"] = file_name
            
            # Add to dataset
            dataset.append(record)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing file {json_file}: {e}")
    
    # Save merged dataset
    output_file_path = os.path.join(json_folder_path, output_path)
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully merged {len(dataset)} JSON files to {output_file_path}")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during processing")
        
        return output_file_path
    
    except Exception as e:
        logger.error(f"Error saving merged dataset: {e}")
        raise
