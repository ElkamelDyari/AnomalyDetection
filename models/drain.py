import pandas as pd
import json
import csv
from tqdm import tqdm
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

# ------------------------ Load and label data ------------------------

def load_log_file(log_file_path, max_lines=None):
    logs, labels = [], []
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            parts = line.strip().split(" ", 1)
            if len(parts) > 1:
                labels.append(0 if parts[0] == "-" else 1)
                logs.append(parts[1])
    return pd.DataFrame({"label": labels, "log": logs})


# ------------------------ Extract structured fields ------------------------

def enrich_log_structure(df):
    fields = ["Timestamp", "Date", "Node", "Time", "NodeRepeat", "Type", "Component", "Level", "Content"]
    parsed_data = {field: [] for field in fields}

    for log_line in df["log"]:
        s = log_line.strip().split()
        if len(s) >= 9:
            for i, field in enumerate(fields[:-1]):
                parsed_data[field].append(s[i])
            parsed_data["Content"].append(" ".join(s[8:]))
        else:
            for field in fields:
                parsed_data[field].append("")

    for field in fields:
        df[field] = parsed_data[field]

    return df


# ------------------------ Drain3 processing (low memory) ------------------------

def init_drain(state_file="drain3_state.bin"):
    persistence = FilePersistence(state_file)
    config = TemplateMinerConfig()
    return TemplateMiner(persistence_handler=persistence, config=config)

def stream_templates(log_lines, template_miner):
    for log in log_lines:
        result = template_miner.add_log_message(log)
        template = result["template_mined"]
        params = {}

        if "<*>" in template:
            tpl_parts = template.split()
            log_parts = log.split()
            idx = 0
            for t, l in zip(tpl_parts, log_parts):
                if t == "<*>":
                    params[f"param_{idx}"] = l
                    idx += 1
        yield template, log, params

def process_log_column(df, column_name, output_csv, batch_size=100_000):
    template_miner = init_drain(state_file=f"{column_name}_state.bin")

    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["template", "log", "params", "label"])  # Include label

        buffer = []
        label_buffer = []
        for i, log in enumerate(tqdm(df[column_name], desc=f"Processing column '{column_name}'")):
            buffer.append(log)
            label_buffer.append(df.iloc[i]["label"])  # Keep labels aligned

            if len(buffer) >= batch_size:
                for (template, _, params), lg, label in zip(stream_templates(buffer, template_miner), buffer, label_buffer):
                    writer.writerow([template, lg, json.dumps(params), label])
                buffer.clear()
                label_buffer.clear()

        if buffer:
            for (template, _, params), lg, label in zip(stream_templates(buffer, template_miner), buffer, label_buffer):
                writer.writerow([template, lg, json.dumps(params), label])


# ------------------------ RUN ------------------------

if __name__ == "__main__":
    df = load_log_file("data/raw/BGL_data.log")
    df = enrich_log_structure(df)
    # Save parsed fields (optional)
    # Apply Drain on the 'Content' field
    process_log_column(df, "Content", output_csv="data/drain/bgl_templates_from_content.csv")
    print("Templates extracted and saved to data/drain/bgl_templates_from_content.csv")
