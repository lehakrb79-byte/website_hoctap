from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
import random
import google.generativeai as genai
from datetime import datetime
import secrets
import PyPDF2
from functools import wraps
import hashlib
from dotenv import load_dotenv

load_dotenv()

# Sử dụng API key từ biến môi trường
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = Flask(__name__)

app.secret_key = 'f8d4e3b2a9c7f1e6d5b8a3c2f9e7d4b1a8c5f2e9d6b3a7f4e1c8b5d2f9a6e3c'

# Thư mục chứa file PDF
PDF_FOLDER = "data"

def load_json(filename):
    try:
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, "r", encoding="utf-8") as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Lỗi đọc file {filename}: {e}")
    except Exception as e:
        print(f"Lỗi: {e}")
    return {}

def save_json(filename, data):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Lỗi lưu file {filename}: {e}")
        return False

def read_pdf(pdf_path):
    """Đọc nội dung từ file PDF"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            # Giới hạn số trang để tránh quá tải (tối đa 50 trang)
            max_pages = min(num_pages, 50)
            
            for page_num in range(max_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        print(f"Lỗi đọc PDF {pdf_path}: {e}")
        return ""

def load_pdfs_by_subject(subject=None):
    """Đọc file PDF theo môn học hoặc tất cả"""
    pdf_contents = {}
    
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Đã tạo thư mục {PDF_FOLDER}")
        return pdf_contents
    
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith('.pdf'):
            # Nếu có subject cụ thể, chỉ đọc file có chứa tên môn đó
            if subject and subject.lower() not in filename.lower():
                continue
                
            pdf_path = os.path.join(PDF_FOLDER, filename)
            content = read_pdf(pdf_path)
            if content:
                pdf_contents[filename] = content
                print(f"Đã đọc file: {filename} ({len(content)} ký tự)")
    
    return pdf_contents

def get_quiz_data_context(subject=None):
    """Lấy dữ liệu câu hỏi theo môn học"""
    context = ""
    data = load_json("data.json")
    
    if data:
        context += "=== DỮ LIỆU CÂU HỎI TRẮC NGHIỆM ===\n\n"
        
        for subj, exams in data.items():
            if subj in ["stem", "players"]:
                continue
            
            # Nếu có subject cụ thể, chỉ lấy môn đó
            if subject and subject.lower() != subj.lower():
                continue
                
            context += f"\nMôn: {subj}\n"
            for exam_name, questions in exams.items():
                if isinstance(questions, list):
                    context += f"  {exam_name}: {len(questions)} câu hỏi\n"
                    # Thêm 2-3 câu hỏi mẫu
                    for i, q in enumerate(questions[:3]):
                        context += f"    - {q.get('question', '')}\n"
        context += "\n"
    
    return context

def clean_markdown(text):
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s*(.*?)(\n|$)', r'\1\n', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    return text

def call_gemini_api(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            cleaned_text = clean_markdown(response.text)
            return cleaned_text
        except Exception as e:
            print(f"API lỗi (lần {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return f"Không thể kết nối với AI. Vui lòng thử lại sau."
    return "Đã xảy ra lỗi không xác định."

def get_subject_prompt(subject):
    """Tạo prompt chuyên biệt cho từng môn học"""
    prompts = {
        "toan": """Bạn là giáo viên Toán lớp 8 chuyên nghiệp với nhiều năm kinh nghiệm giảng dạy.
        
Chuyên môn của bạn:
- Đại số: Phương trình, bất phương trình, hàm số
- Hình học: Định lý Pythagoras, tam giác đồng dạng, diện tích, thể tích
- Toán thực tế và ứng dụng
- Xưng cô

Phong cách dạy:
- Giải thích từng bước một cách logic và dễ hiểu
- Sử dụng sơ đồ, hình vẽ minh họa khi cần
- Đưa ra nhiều ví dụ từ dễ đến khó
- Hướng dẫn nhiều cách giải khác nhau
- Xưng cô
- Khuyến khích tư duy logic và sáng tạo""",

        "ly": """Bạn là giáo viên Vật Lý lớp 8 đầy nhiệt huyết và am hiểu sâu sắc.

Chuyên môn của bạn:
- Cơ học: Chuyển động, lực, áp suất
- Nhiệt học: Nhiệt độ, nhiệt lượng, sự truyền nhiệt
- Điện học: Mạch điện, định luật Ohm, công suất điện
- Quang học: Phản xạ, khúc xạ ánh sáng, thấu kính

Phong cách dạy:
- Kết nối lý thuyết với hiện tượng thực tế trong đời sống
- Giải thích bằng thí nghiệm và mô phỏng
- Phân tích công thức và đại lượng vật lý
- Hướng dẫn cách vẽ sơ đồ, mạch điện, đường truyền ánh sáng
- Xưng cô
- Nhấn mạnh an toàn trong thí nghiệm""",

        "hoa": """Bạn là giáo viên Hóa học lớp 8 tận tâm và giàu kinh nghiệm.

Chuyên môn của bạn:
- Nguyên tử, phân tử, nguyên tố hóa học
- Phản ứng hóa học: Phân loại, cân bằng phương trình
- Axit, bazơ, muối: Tính chất, ứng dụng
- Bảng tuần hoàn các nguyên tố

Phong cách dạy:
- Giải thích cấu trúc phân tử một cách trực quan
- Hướng dẫn viết và cân bằng phương trình hóa học
- Kết nối với ứng dụng thực tế trong đời sống
- Nhấn mạnh an toàn hóa chất
- Xưng cô
- Sử dụng ví dụ từ thiên nhiên và công nghiệp""",

        "sinh": """Bạn là giáo viên Sinh học lớp 8 yêu thiên nhiên và đam mê giảng dạy.

Chuyên môn của bạn:
- Tế bào học: Cấu tạo, chức năng của tế bào
- Sinh lý người: Hệ tuần hoàn, hô hấp, tiêu hóa, thần kinh
- Sinh thái học: Mối quan hệ sinh vật - môi trường
- Di truyền học cơ bản

Phong cách dạy:
- Sử dụng hình ảnh minh họa sinh động
- Kết nối với sức khỏe và đời sống hàng ngày
- Giải thích các quá trình sinh học một cách dễ hiểu
- Khuyến khích quan sát và khám phá tự nhiên
- Xưng cô
- Nhấn mạnh bảo vệ môi trường và sức khỏe"""
    }
    
    return prompts.get(subject, "")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot_home():
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        subject = request.form.get('subject', '').strip()
        
        if not question:
            return jsonify({'error': 'Vui lòng nhập câu hỏi!', 'answer': None}), 400
        
        if not subject:
            return jsonify({'error': 'Vui lòng chọn môn học!', 'answer': None}), 400
        
        try:
            # Đọc PDF theo môn học
            if subject == 'tong':
                pdf_contents = load_pdfs_by_subject()
                quiz_context = get_quiz_data_context()
                subject_name = "tất cả các môn"
                subject_prompt = "Bạn là trợ lý học tập đa năng, giỏi tất cả các môn Toán, Lý, Hóa, Sinh lớp 8."
            else:
                pdf_contents = load_pdfs_by_subject(subject)
                quiz_context = get_quiz_data_context(subject)
                subject_names = {'toan': 'Toán', 'ly': 'Vật Lý', 'hoa': 'Hóa học', 'sinh': 'Sinh học'}
                subject_name = subject_names.get(subject, subject)
                subject_prompt = get_subject_prompt(subject)
            
            pdf_count = len(pdf_contents)
            
            # Xây dựng context từ PDF
            pdf_context = ""
            if pdf_contents:
                pdf_context = f"=== TÀI LIỆU HỌC TẬP {subject_name.upper()} (PDF) ===\n\n"
                for filename, content in pdf_contents.items():
                    max_chars = 8000 if subject == 'tong' else 10000
                    truncated_content = content[:max_chars]
                    if len(content) > max_chars:
                        truncated_content += "\n... (nội dung còn lại đã bị cắt bớt)"
                    
                    pdf_context += f"[File: {filename}]\n{truncated_content}\n\n"
            
            # Tạo prompt
            prompt = f"""{subject_prompt}

{quiz_context}

{pdf_context}

Nhiệm vụ của bạn:
1. Phân tích câu hỏi: "{question}"
2. Sử dụng tài liệu PDF và dữ liệu câu hỏi để trả lời chính xác
3. Hướng dẫn từng bước dễ hiểu cho học sinh lớp 8
4. Giải thích lý thuyết liên quan
5. Đưa ra ví dụ minh họa thực tế
6. Khuyến khích tư duy và tự giải quyết

Lưu ý:
- Trả lời bằng tiếng Việt phù hợp học sinh lớp 8
- Trích dẫn cụ thể nếu có trong PDF
- Luôn kiên nhẫn, nhiệt tình và động viên

Hãy trả lời chi tiết, chuyên nghiệp và thân thiện!"""
            
            answer = call_gemini_api(prompt)
            
            # Trả về JSON
            return jsonify({
                'success': True,
                'answer': answer,
                'pdf_count': pdf_count,
                'subject': subject
            })
            
        except Exception as e:
            print(f"Lỗi xử lý chatbot: {e}")
            return jsonify({
                'success': False,
                'error': 'Đã xảy ra lỗi khi xử lý câu hỏi',
                'answer': None
            }), 500
    
    # GET request - render trang chatbot
    return render_template('chatbot.html')
    ###################3
@app.route('/chatbot/<subject>', methods=['GET', 'POST'])
def chatbot(subject):
    """Chatbot chuyên môn theo từng môn học"""
    valid_subjects = ['toan', 'ly', 'hoa', 'sinh', 'tong']
    
    if subject not in valid_subjects:
        flash("Môn học không hợp lệ!", "error")
        return redirect(url_for('chatbot_home'))
    
    answer = ""
    pdf_count = 0
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        
        if not question:
            answer = "Vui lòng nhập câu hỏi!"
        else:
            # Đọc PDF theo môn học (hoặc tất cả nếu là chat tổng)
            if subject == 'tong':
                pdf_contents = load_pdfs_by_subject()
                quiz_context = get_quiz_data_context()
                subject_name = "tất cả các môn"
                subject_prompt = "Bạn là trợ lý học tập đa năng, giỏi tất cả các môn Toán, Lý, Hóa, Sinh lớp 8."
            else:
                pdf_contents = load_pdfs_by_subject(subject)
                quiz_context = get_quiz_data_context(subject)
                subject_names = {'toan': 'Toán', 'ly': 'Vật Lý', 'hoa': 'Hóa học', 'sinh': 'Sinh học'}
                subject_name = subject_names.get(subject, subject)
                subject_prompt = get_subject_prompt(subject)
            
            pdf_count = len(pdf_contents)
            
            # Xây dựng context từ PDF
            pdf_context = ""
            if pdf_contents:
                pdf_context = f"=== TÀI LIỆU HỌC TẬP {subject_name.upper()} (PDF) ===\n\n"
                for filename, content in pdf_contents.items():
                    # Giới hạn độ dài mỗi PDF
                    max_chars = 8000 if subject == 'tong' else 10000
                    truncated_content = content[:max_chars]
                    if len(content) > max_chars:
                        truncated_content += "\n... (nội dung còn lại đã bị cắt bớt)"
                    
                    pdf_context += f"[File: {filename}]\n{truncated_content}\n\n"
            
            # Tạo prompt
            prompt = f"""{subject_prompt}

{quiz_context}

{pdf_context}

Nhiệm vụ của bạn:
1. Phân tích câu hỏi của học sinh: "{question}"
2. Sử dụng tài liệu PDF và dữ liệu câu hỏi trắc nghiệm để trả lời chính xác
3. Hướng dẫn cách giải từng bước một cách dễ hiểu cho học sinh lớp 8
4. Giải thích lý thuyết liên quan một cách sinh động
5. Đưa ra ví dụ minh họa thực tế nếu phù hợp
6. Khuyến khích học sinh tư duy và tự giải quyết vấn đề

Lưu ý quan trọng:
- Trả lời bằng tiếng Việt với ngôn ngữ phù hợp học sinh lớp 8
- Nếu câu hỏi liên quan đến nội dung trong PDF, hãy trích dẫn cụ thể
- Nếu không tìm thấy trong tài liệu, sử dụng kiến thức chuyên môn của bạn
- Luôn kiên nhẫn, nhiệt tình và động viên học sinh
- Với môn Toán: Giải thích từng bước, vẽ sơ đồ nếu cần
- Với môn Lý: Liên hệ thực tế, giải thích hiện tượng
- Với môn Hóa: Viết phương trình, giải thích phản ứng
- Với môn Sinh: Mô tả quá trình, liên hệ sức khỏe

Hãy trả lời câu hỏi một cách chi tiết, chuyên nghiệp và thân thiện!"""
            
            answer = call_gemini_api(prompt)
            
            # Thêm thông tin về tài liệu đã tham khảo
            if pdf_contents:
                answer += f"\n\n---\n📚 Đã tham khảo {pdf_count} file tài liệu {subject_name}"
    
    subject_names = {
        'toan': 'Toán học',
        'ly': 'Vật Lý', 
        'hoa': 'Hóa học',
        'sinh': 'Sinh học',
        'tong': 'Tổng hợp'
    }
    
    return render_template('chatbot_subject.html', 
                         subject=subject,
                         subject_name=subject_names.get(subject, subject),
                         answer=answer,
                         pdf_count=pdf_count)

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        subject = request.form.get("subject", "").strip()
        
        if not name or not subject:
            flash("Vui lòng nhập đầy đủ thông tin!", "warning")
            return redirect(url_for("quiz"))
        
        session["player_name"] = name
        session["subject"] = subject
        return redirect(url_for("play"))
    
    data = load_json("game.json")
    subjects = list(data.keys())
    subjects = [s for s in subjects if s not in ["stem", "players"]]
    
    return render_template("quiz.html", subjects=subjects)

@app.route("/play")
def play():
    subject = session.get("subject")
    name = session.get("player_name")
    
    if not subject or not name:
        flash("Vui lòng chọn môn học!", "warning")
        return redirect(url_for("quiz"))
    
    data = load_json("game.json")
    
    if subject not in data:
        flash(f"Không tìm thấy môn {subject}!", "error")
        return redirect(url_for("quiz"))
    
    questions = []
    for exam_name, exam_questions in data[subject].items():
        if isinstance(exam_questions, list):
            questions.extend(exam_questions)
    
    if not questions:
        flash("Không có câu hỏi nào!", "error")
        return redirect(url_for("quiz"))
    
    random.shuffle(questions)
    questions = questions[:10]
    
    session["questions"] = questions
    session["index"] = 0
    session["score"] = 0
    session["start_time"] = datetime.now().isoformat()
    
    return render_template("play.html", 
                         question=questions[0], 
                         index=0, 
                         total=len(questions))

@app.route("/next", methods=["POST"])
def next_question():
    answer = request.form.get("answer")
    index = session.get("index", 0)
    questions = session.get("questions", [])
    score = session.get("score", 0)
    
    if not questions:
        return redirect(url_for("quiz"))
    
    if answer and index < len(questions):
        if answer == questions[index]["answer"]:
            score += 1
    
    index += 1
    session["score"] = score
    session["index"] = index
    
    if index >= len(questions):
        name = session.get("player_name", "Người chơi")
        subject = session.get("subject", "Unknown")
        
        start_time = session.get("start_time")
        duration = 0
        if start_time:
            try:
                start = datetime.fromisoformat(start_time)
                duration = int((datetime.now() - start).total_seconds())
            except:
                pass
        
        data = load_json("game.json")
        if "players" not in data:
            data["players"] = []
        
        data["players"].append({
            "name": name,
            "subject": subject,
            "score": score,
            "total": len(questions),
            "duration": duration,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        save_json("game.json", data)
        
        session.pop("questions", None)
        session.pop("index", None)
        session.pop("start_time", None)
        
        return redirect(url_for("leaderboard"))
    
    return render_template("play.html", 
                         question=questions[index], 
                         index=index, 
                         total=len(questions))

@app.route("/leaderboard")
def leaderboard():
    data = load_json("game.json")
    players = data.get("players", [])
    
    players_sorted = sorted(
        players, 
        key=lambda x: (-x.get("score", 0), x.get("duration", 999999))
    )
    
    top_players = players_sorted[:10]
    
    for i, player in enumerate(top_players, 1):
        player["rank"] = i
    
    current_score = session.get("score")
    current_total = len(session.get("questions", []))
    
    return render_template("leaderboard.html", 
                         players=top_players,
                         current_score=current_score,
                         current_total=current_total)
########################################################################################################3
@app.route('/materials')
def materials():
    materials_data = {
        "Lý": [
            {
                "title": "Cơ học - Chuyển động",
                "type": "pdf",
                "link": "https://drive.google.com/file/d/1DEF456/preview",
                "description": "Lý thuyết chuyển động đều và biến đổi đều"
            },
            {
                "title": "Điện học cơ bản",
                "type": "video",
                "link": "https://www.youtube.com/embed/RGfQuH2egDlgAOwo",
                "description": "Video giải thích định luật Ohm"
            },
            {
                "title": "Quang học - Gương phẳng",
                "type": "pdf",
                "link": "https://drive.google.com/file/d/1GHI012/preview",
                "description": "Bài giảng về gương phẳng và phản xạ ánh sáng"
            }
        ],
        "Hóa": [
            {
                "title": "Bảng tuần hoàn",
                "type": "pdf",
                "link": "https://drive.google.com/file/d/1JKL345/preview",
                "description": "Bảng tuần hoàn các nguyên tố hóa học"
            },
            {
                "title": "Phản ứng hóa học",
                "type": "video",
                "link": "https://www.youtube.com/embed/VIDEO_ID2",
                "description": "Các loại phản ứng hóa học cơ bản"
            },
            {
                "title": "Axit - Bazơ - Muối",
                "type": "pdf",
                "link": "https://drive.google.com/file/d/1MNO678/preview",
                "description": "Lý thuyết và bài tập về axit, bazơ, muối"
            }
        ],
        "Sinh": [
            {
                "title": "Tế bào học",
                "type": "pdf",
                "link": "https://drive.google.com/file/d/1PQR901/preview",
                "description": "Cấu tạo và chức năng của tế bào"
            },
            {
                "title": "Quang hợp ở thực vật",
                "type": "video",
                "link": "https://www.youtube.com/embed/VIDEO_ID3",
                "description": "Video minh họa quá trình quang hợp"
            },
            {
                "title": "Hệ cơ quan người",
                "type": "pdf",
                "link": "https://drive.google.com/file/d/1STU234/preview",
                "description": "Các hệ cơ quan trong cơ thể người"
            }
        ]
    }
    return render_template('materials.html', materials=materials_data)

@app.route('/test', methods=['GET', 'POST'])
def test():
    data = load_json("data.json")
    subject = request.args.get('subject')
    exam = request.args.get('exam')
    result = None
    
    if not subject:
        subjects = [k for k in data.keys() if k not in ["stem", "players"]]
        return render_template('test.html', step='subject', subjects=subjects)
    
    elif subject and not exam:
        if subject not in data:
            flash(f"Không tìm thấy môn {subject}!", "error")
            return redirect(url_for('test'))
        exams = list(data[subject].keys())
        return render_template('test.html', step='exam', subject=subject, exams=exams)
    
    else:
        if subject not in data or exam not in data[subject]:
            flash("Không tìm thấy đề thi!", "error")
            return redirect(url_for('test'))
        
        questions = data[subject][exam]
        
        if request.method == 'POST':
            correct = 0
            wrong_answers = []
            
            for i, q in enumerate(questions):
                user_answer = request.form.get(f"q{i}", "").strip().upper()
                correct_answer = q.get("answer", "")
                
                if user_answer == correct_answer:
                    correct += 1
                else:
                    # Lấy giải thích nếu có, nếu không thì tự động tạo
                    explanation = q.get("explanation", "")
                    
                    # Nếu không có giải thích trong data, tạo giải thích bằng AI
                    if not explanation:
                        question_text = q.get("question", "")
                        options = {
                            "A": q.get("A", ""),
                            "B": q.get("B", ""),
                            "C": q.get("C", ""),
                            "D": q.get("D", "")
                        }
                        correct_option = options.get(correct_answer, "")
                        
                        prompt = f"""Bạn là giáo viên giỏi. Hãy giải thích ngắn gọn (2-3 câu) tại sao đáp án {correct_answer} ({correct_option}) là đúng cho câu hỏi sau:

Câu hỏi: {question_text}
Các đáp án:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

Đáp án đúng: {correct_answer}. {correct_option}

Giải thích bằng tiếng Việt, dễ hiểu cho học sinh lớp 8."""
                        
                        explanation = call_gemini_api(prompt)
                    
                    wrong_answers.append({
                        "question": q["question"],
                        "your_answer": user_answer or "Không trả lời",
                        "correct_answer": correct_answer,
                        "explanation": explanation,
                        "A": q.get("A", ""),
                        "B": q.get("B", ""),
                        "C": q.get("C", ""),
                        "D": q.get("D", "")
                    })
            
            total = len(questions)
            percentage = (correct / total * 100) if total > 0 else 0
            
            result = {
                "correct": correct,
                "total": total,
                "percentage": round(percentage, 1),
                "wrong_answers": wrong_answers
            }
        
        return render_template('test.html', 
                             step='quiz', 
                             subject=subject, 
                             exam=exam, 
                             questions=questions, 
                             result=result)

@app.route('/advisor', methods=['GET', 'POST'])
def advisor():
    plan = ""
    if request.method == 'POST':
        info = request.form.get('info', '').strip()
        if not info:
            plan = "Vui lòng nhập thông tin của bạn!"
        else:
            prompt = f"""Bạn là cố vấn học tập chuyên nghiệp. Hãy tạo lộ trình học tập chi tiết và phù hợp cho học sinh có đặc điểm sau:

{info}

Lộ trình cần bao gồm:
1. Đánh giá điểm mạnh/yếu
2. Mục tiêu học tập cụ thể
3. Kế hoạch học từng môn (thời gian, phương pháp)
4. Lời khuyên và động viên

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng."""
            
            plan = call_gemini_api(prompt)
    
    return render_template('advisor.html', plan=plan)

@app.route('/stem', methods=['GET', 'POST'])
def stem():
    experiments = {
        "Lý": [
            {
                "id": "circuit_construction",
                "title": "Mạch điện AC/DC",
                "category": "Điện học",
                "desc": "Xây dựng và thí nghiệm với các mạch điện, đo điện áp, dòng điện",
                "difficulty": "Trung bình",
                "phet_url": "https://phet.colorado.edu/sims/html/circuit-construction-kit-ac-virtual-lab/latest/circuit-construction-kit-ac-virtual-lab_all.html"
            },
            {
                "id": "geometric_optics",
                "title": "Quang học hình học",
                "category": "Quang học",
                "desc": "Nghiên cứu thấu kính, gương và sự tạo ảnh",
                "difficulty": "Trung bình",
                "phet_url": "https://phet.colorado.edu/sims/html/geometric-optics/latest/geometric-optics_all.html"
            },
            {
                "id": "energy_forms",
                "title": "Năng lượng và chuyển hóa",
                "category": "Nhiệt học",
                "desc": "Khám phá các dạng năng lượng và sự chuyển hóa giữa chúng",
                "difficulty": "Trung bình",
                "phet_url": "https://phet.colorado.edu/sims/html/energy-forms-and-changes/latest/energy-forms-and-changes_all.html"
            },
            {
                "id": "static_electricity",
                "title": "Điện tích và điện tĩnh",
                "category": "Điện học",
                "desc": "Tìm hiểu về điện tích, lực tĩnh điện và hiện tượng nhiễm điện",
                "difficulty": "Dễ",
                "phet_url": "https://phet.colorado.edu/sims/html/balloons-and-static-electricity/latest/balloons-and-static-electricity_all.html"
            },
            {
                "id": "friction",
                "title": "Lực ma sát",
                "category": "Cơ học",
                "desc": "Nghiên cứu lực ma sát giữa các bề mặt khác nhau",
                "difficulty": "Dễ",
                "phet_url": "https://phet.colorado.edu/sims/html/friction/latest/friction_all.html"
            },
            {
                "id": "forces_motion",
                "title": "Lực và chuyển động",
                "category": "Cơ học",
                "desc": "Thí nghiệm về lực, gia tốc và định luật Newton",
                "difficulty": "Trung bình",
                "phet_url": "https://phet.colorado.edu/sims/html/forces-and-motion-basics/latest/forces-and-motion-basics_all.html"
            }
        ],
       "Hóa": [
            {
                "id": "build_molecule",
                "title": "Xây dựng phân tử",
                "category": "Hóa học phân tử",
                "desc": "Tạo các phân tử từ nguyên tử, tìm hiểu về liên kết hóa học",
                "difficulty": "Dễ",
                "phet_url": "https://phet.colorado.edu/sims/html/build-a-molecule/latest/build-a-molecule_all.html?locale=vi"
            },
            {
                "id": "build_atom",
                "title": "Xây dựng nguyên tử",
                "category": "Cấu tạo nguyên tử",
                "desc": "Tìm hiểu cấu tạo nguyên tử, proton, neutron và electron",
                "difficulty": "Dễ",
                "phet_url": "https://phet.colorado.edu/sims/html/build-an-atom/latest/build-an-atom_all.html?locale=vi"
            }
        ],
        "Sinh": [
            {
                "id": "gene_expression",
                "title": "Biểu hiện gen",
                "category": "Di truyền học",
                "desc": "Tìm hiểu cách gen điều khiển tổng hợp protein",
                "difficulty": "Khó",
                "phet_url": "https://phet.colorado.edu/sims/html/gene-expression-essentials/latest/gene-expression-essentials_all.html"
            }
        ]
    }
    
    selected_subject = request.args.get('subject', 'all')
    
    if selected_subject != 'all' and selected_subject in experiments:
        filtered_experiments = {selected_subject: experiments[selected_subject]}
    else:
        filtered_experiments = experiments
    
    subjects = list(experiments.keys())
    
    return render_template('stem.html', 
                         experiments=filtered_experiments,
                         subjects=subjects,
                         selected_subject=selected_subject)
                         ##################
@app.route('/stem/experiment/<exp_id>')
def experiment_detail(exp_id):
    experiments_map = {
        # VẬT LÝ
        "circuit_construction": {
            "title": "Mạch điện AC/DC",
            "subject": "Lý",
            "category": "Điện học",
            "desc": "Xây dựng và thí nghiệm với các mạch điện AC/DC, đo điện áp, dòng điện",
            "phet_url": "https://phet.colorado.edu/sims/html/circuit-construction-kit-ac-virtual-lab/latest/circuit-construction-kit-ac-virtual-lab_all.html",
            "instructions": [
                "Kéo thả các linh kiện điện (pin, bóng đèn, điện trở, công tắc) từ thanh công cụ",
                "Nối các linh kiện bằng dây dẫn để tạo thành mạch điện",
                "Sử dụng đồng hồ đo để đo hiệu điện thế và cường độ dòng điện",
                "Thử nghiệm mạch nối tiếp và mạch song song",
                "Quan sát sự thay đổi khi thêm hoặc bớt linh kiện"
            ]
        },
        "geometric_optics": {
            "title": "Quang học hình học",
            "subject": "Lý",
            "category": "Quang học",
            "desc": "Nghiên cứu thấu kính, gương và sự tạo ảnh trong quang học hình học",
            "phet_url": "https://phet.colorado.edu/sims/html/geometric-optics/latest/geometric-optics_all.html",
            "instructions": [
                "Chọn loại thấu kính hoặc gương (hội tụ, phân kỳ)",
                "Di chuyển vật để quan sát sự thay đổi của ảnh",
                "Đo khoảng cách vật, ảnh và tiêu cự",
                "Quan sát đường truyền của tia sáng",
                "So sánh ảnh thật và ảnh ảo"
            ]
        },
        "energy_forms": {
            "title": "Năng lượng và chuyển hóa",
            "subject": "Lý",
            "category": "Nhiệt học",
            "desc": "Khám phá các dạng năng lượng và sự chuyển hóa giữa chúng",
            "phet_url": "https://phet.colorado.edu/sims/html/energy-forms-and-changes/latest/energy-forms-and-changes_all.html",
            "instructions": [
                "Quan sát các dạng năng lượng: nhiệt, cơ, điện, hóa, ánh sáng",
                "Thả các vật thể vào nước và quan sát sự trao đổi nhiệt",
                "Sử dụng bếp đun để gia nhiệt cho nước",
                "Theo dõi biểu đồ năng lượng trong quá trình chuyển hóa",
                "Thí nghiệm với các vật liệu cách nhiệt khác nhau"
            ]
        },
        "static_electricity": {
            "title": "Điện tích và điện tĩnh",
            "subject": "Lý",
            "category": "Điện học",
            "desc": "Tìm hiểu về điện tích, lực tĩnh điện và hiện tượng nhiễm điện",
            "phet_url": "https://phet.colorado.edu/sims/html/balloons-and-static-electricity/latest/balloons-and-static-electricity_all.html",
            "instructions": [
                "Chà xát quả bóng bay vào áo len",
                "Quan sát sự chuyển dịch điện tích",
                "Đưa quả bóng lại gần tường và quan sát",
                "Chú ý lực hút giữa các điện tích trái dấu",
                "Thí nghiệm với nhiều quả bóng bay"
            ]
        },
        "friction": {
            "title": "Lực ma sát",
            "subject": "Lý",
            "category": "Cơ học",
            "desc": "Nghiên cứu lực ma sát giữa các bề mặt khác nhau",
            "phet_url": "https://phet.colorado.edu/sims/html/friction/latest/friction_all.html",
            "instructions": [
                "Kéo sách trên các bề mặt khác nhau",
                "Quan sát nhiệt độ tăng do ma sát",
                "Thay đổi lực kéo và quan sát chuyển động",
                "So sánh ma sát giữa các bề mặt: gỗ, băng",
                "Quan sát các phân tử ở mức vi mô"
            ]
        },
        "forces_motion": {
            "title": "Lực và chuyển động",
            "subject": "Lý",
            "category": "Cơ học",
            "desc": "Thí nghiệm về lực, gia tốc và định luật Newton",
            "phet_url": "https://phet.colorado.edu/sims/html/forces-and-motion-basics/latest/forces-and-motion-basics_all.html",
            "instructions": [
                "Đẩy và kéo vật thể để quan sát chuyển động",
                "Thay đổi lực tác dụng và khối lượng",
                "Quan sát gia tốc thay đổi theo lực",
                "Thí nghiệm với ma sát và không ma sát",
                "Áp dụng định luật Newton"
            ]
        },
        
        # HÓA HỌC
        "build_atom": {
            "title": "Xây dựng nguyên tử",
            "subject": "Hóa",
            "category": "Cấu tạo nguyên tử",
            "desc": "Tạo nguyên tử từ proton, neutron và electron",
            "phet_url": "https://phet.colorado.edu/sims/html/build-an-atom/latest/build-an-atom_all.html",
            "instructions": [
                "Kéo proton, neutron vào hạt nhân",
                "Thêm electron vào các lớp vỏ",
                "Quan sát ký hiệu hóa học và số khối",
                "Tạo các nguyên tử và ion khác nhau",
                "Kiểm tra điện tích tổng của nguyên tử"
            ]
        },
        "molecule_shapes": {
            "title": "Hình dạng phân tử",
            "subject": "Hóa",
            "category": "Liên kết hóa học",
            "desc": "Khám phá hình dạng và cấu trúc của các phân tử",
            "phet_url": "https://phet.colorado.edu/sims/html/molecule-shapes/latest/molecule-shapes_all.html",
            "instructions": [
                "Chọn nguyên tử trung tâm",
                "Thêm các nguyên tử xung quanh",
                "Quan sát hình dạng phân tử 3D",
                "Thay đổi số lượng liên kết và cặp electron",
                "Học về góc liên kết và lực đẩy"
            ]
        },
        "ph_scale": {
            "title": "Thang đo pH",
            "subject": "Hóa",
            "category": "Axit - Bazơ",
            "desc": "Đo pH của các dung dịch axit, bazơ và trung tính",
            "phet_url": "https://phet.colorado.edu/sims/html/ph-scale/latest/ph-scale_all.html",
            "instructions": [
                "Chọn các dung dịch khác nhau để đo pH",
                "Sử dụng giấy quỳ hoặc máy đo pH",
                "Quan sát màu sắc thay đổi theo pH",
                "So sánh nồng độ H+ và OH-",
                "Phân loại axit mạnh, yếu, bazơ mạnh, yếu"
            ]
        },
        "acid_base_solutions": {
            "title": "Dung dịch axit-bazơ",
            "subject": "Hóa",
            "category": "Axit - Bazơ",
            "desc": "Nghiên cứu tính chất của dung dịch axit và bazơ",
            "phet_url": "https://phet.colorado.edu/sims/html/acid-base-solutions/latest/acid-base-solutions_all.html",
            "instructions": [
                "Tạo dung dịch axit và bazơ với nồng độ khác nhau",
                "Quan sát độ phân ly trong dung dịch",
                "So sánh axit mạnh và axit yếu",
                "Đo pH và nồng độ ion",
                "Thí nghiệm với các chỉ thị màu"
            ]
        },
        "reactants_products": {
            "title": "Chất phản ứng và sản phẩm",
            "subject": "Hóa",
            "category": "Phản ứng hóa học",
            "desc": "Quan sát và cân bằng các phản ứng hóa học",
            "phet_url": "https://phet.colorado.edu/sims/html/reactants-products-and-leftovers/latest/reactants-products-and-leftovers_all.html",
            "instructions": [
                "Chọn phản ứng hóa học",
                "Thêm chất phản ứng với tỷ lệ khác nhau",
                "Quan sát sản phẩm được tạo ra",
                "Xác định chất dư và chất hết",
                "Cân bằng phương trình hóa học"
            ]
        },
        "states_matter": {
            "title": "Trạng thái của vật chất",
            "subject": "Hóa",
            "category": "Trạng thái vật chất",
            "desc": "Khám phá rắn, lỏng, khí và chuyển đổi trạng thái",
            "phet_url": "https://phet.colorado.edu/sims/html/states-of-matter/latest/states-of-matter_all.html",
            "instructions": [
                "Điều chỉnh nhiệt độ và áp suất",
                "Quan sát chuyển động của phân tử",
                "Xem quá trình nóng chảy, đông đặc",
                "Quan sát sự bay hơi và ngưng tụ",
                "So sánh các chất khác nhau"
            ]
        },
        
        # SINH HỌC
        "gene_expression": {
            "title": "Biểu hiện gen",
            "subject": "Sinh",
            "category": "Di truyền học",
            "desc": "Tìm hiểu cách gen điều khiển tổng hợp protein",
            "phet_url": "https://phet.colorado.edu/sims/html/gene-expression-essentials/latest/gene-expression-essentials_all.html",
            "instructions": [
                "Quan sát quá trình phiên mã DNA → RNA",
                "Theo dõi quá trình dịch mã RNA → protein",
                "Điều chỉnh mức độ biểu hiện gen",
                "Quan sát ảnh hưởng của đột biến",
                "Tìm hiểu về điều hòa gen"
            ]
        },
        "natural_selection": {
            "title": "Chọn lọc tự nhiên",
            "subject": "Sinh",
            "category": "Tiến hóa",
            "desc": "Mô phỏng quá trình chọn lọc tự nhiên",
            "phet_url": "https://phet.colorado.edu/sims/html/natural-selection/latest/natural-selection_all.html",
            "instructions": [
                "Chọn đặc điểm của quần thể thỏ",
                "Thay đổi môi trường sống",
                "Quan sát sự thay đổi tần số gen",
                "Theo dõi số lượng cá thể qua các thế hệ",
                "Phân tích vai trò của đột biến và môi trường"
            ]
        },
        "neuron": {
            "title": "Tế bào thần kinh",
            "subject": "Sinh",
            "category": "Sinh lý người",
            "desc": "Khám phá cách tế bào thần kinh truyền tín hiệu",
            "phet_url": "https://phet.colorado.edu/sims/html/neuron/latest/neuron_all.html",
            "instructions": [
                "Quan sát cấu trúc tế bào thần kinh",
                "Kích thích tế bào và xem xung thần kinh",
                "Theo dõi dòng ion qua màng tế bào",
                "Quan sát điện thế màng thay đổi",
                "Tìm hiểu về synapse và truyền tín hiệu"
            ]
        },
        "build_molecule": {
            "title": "Xây dựng phân tử sinh học",
            "subject": "Sinh",
            "category": "Sinh hóa",
            "desc": "Tạo các phân tử hữu cơ và sinh học",
            "phet_url": "https://phet.colorado.edu/sims/html/build-a-molecule/latest/build-a-molecule_all.html",
            "instructions": [
                "Kéo các nguyên tử để tạo phân tử",
                "Tạo các phân tử đơn giản (H2O, CO2)",
                "Xây dựng phân tử hữu cơ phức tạp",
                "Kiểm tra công thức phân tử",
                "Học về liên kết hóa học trong sinh học"
            ]
        },
        "cell_structure": {
            "title": "Cấu trúc tế bào",
            "subject": "Sinh",
            "category": "Tế bào học",
            "desc": "Khám phá cấu trúc và chức năng của tế bào",
            "phet_url": "https://phet.colorado.edu/sims/html/cell-structure/latest/cell-structure_all.html",
            "instructions": [
                "Quan sát các bào quan trong tế bào",
                "Tìm hiểu chức năng của từng bào quan",
                "So sánh tế bào động vật và thực vật",
                "Quan sát màng tế bào và vận chuyển",
                "Học về ti thể, lục lạp, nhân tế bào"
            ]
        },
        "biomolecules": {
            "title": "Phân tử sinh học",
            "subject": "Sinh",
            "category": "Sinh hóa",
            "desc": "Tìm hiểu về protein, lipid, carbohydrate và DNA",
            "phet_url": "https://phet.colorado.edu/sims/html/biomolecules/latest/biomolecules_all.html",
            "instructions": [
                "Khám phá cấu trúc các đại phân tử",
                "Tìm hiểu về protein và amino acid",
                "Quan sát cấu trúc DNA và RNA",
                "Học về carbohydrate và lipid",
                "Phân tích chức năng của từng loại phân tử"
            ]
        }
    }
    
    experiment = experiments_map.get(exp_id)
    if not experiment:
        flash("Thí nghiệm không tồn tại!", "error")
        return redirect(url_for('stem'))
    
    return render_template('experiment.html', experiment=experiment, exp_id=exp_id)
#####################################
# ===============================================
# PHẦN 1: IMPORT VÀ HELPER FUNCTIONS BỔ SUNG
# ===============================================

from functools import wraps
import hashlib

# Decorator kiểm tra đăng nhập
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Vui lòng đăng nhập!", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Decorator kiểm tra quyền giáo viên
def teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Vui lòng đăng nhập!", "warning")
            return redirect(url_for('login'))
        if session.get('role') != 'teacher':
            flash("Bạn không có quyền truy cập!", "error")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Hàm hash mật khẩu đơn giản
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Khởi tạo file users.json nếu chưa có
def init_users():
    if not os.path.exists("users.json"):
        users_data = {
            "teachers": [
                {
                    "id": "gv001",
                    "username": "giaovien1",
                    "password": hash_password("123456"),
                    "fullname": "Cô Nguyễn Thị A",
                    "email": "coA@school.edu.vn",
                    "subject": "toan"
                },
                {
                    "id": "gv002",
                    "username": "giaovien2",
                    "password": hash_password("123456"),
                    "fullname": "Thầy Trần Văn B",
                    "email": "thayB@school.edu.vn",
                    "subject": "ly"
                }
            ],
            "students": []
        }
        save_json("users.json", users_data)

# Khởi tạo file courses.json
def init_courses():
    if not os.path.exists("courses.json"):
        courses_data = {
            "courses": [],
            "lessons": [],
            "student_progress": []
        }
        save_json("courses.json", courses_data)

# ===============================================
# PHẦN 2: AUTHENTICATION ROUTES
# ===============================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash("Vui lòng nhập đầy đủ thông tin!", "warning")
            return redirect(url_for('login'))
        
        users_data = load_json("users.json")
        
        # Kiểm tra giáo viên
        for teacher in users_data.get('teachers', []):
            # Kiểm tra cả 2 trường hợp: password thô HOẶC password đã hash
            teacher_password = teacher['password']
            
            # Case 1: Password thô (như "testgv1", "coleha@")
            # Case 2: Password đã hash (dài 64 ký tự)
            if teacher['username'] == username:
                # Nếu password trong DB là thô (không phải hash 64 ký tự)
                if len(teacher_password) < 64 and teacher_password == password:
                    # Đăng nhập với password thô
                    session['user_id'] = teacher['id']
                    session['username'] = teacher['username']
                    session['fullname'] = teacher['fullname']
                    session['role'] = 'teacher'
                    session['subject'] = teacher.get('subject', '')
                    flash(f"Chào mừng {teacher['fullname']}!", "success")
                    return redirect(url_for('teacher_dashboard'))
                # Nếu password trong DB là hash (64 ký tự)
                elif len(teacher_password) == 64 and teacher_password == hash_password(password):
                    # Đăng nhập với password đã hash
                    session['user_id'] = teacher['id']
                    session['username'] = teacher['username']
                    session['fullname'] = teacher['fullname']
                    session['role'] = 'teacher'
                    session['subject'] = teacher.get('subject', '')
                    flash(f"Chào mừng {teacher['fullname']}!", "success")
                    return redirect(url_for('teacher_dashboard'))
        
        # Kiểm tra học sinh - LUÔN DÙNG HASH
        hashed_pw = hash_password(password)
        for student in users_data.get('students', []):
            if student['username'] == username:
                # Kiểm tra cả password thô và hash
                if student['password'] == hashed_pw or student['password'] == password:
                    session['user_id'] = student['id']
                    session['username'] = student['username']
                    session['fullname'] = student['fullname']
                    session['role'] = 'student'
                    flash(f"Chào mừng {student['fullname']}!", "success")
                    return redirect(url_for('student_dashboard'))
        
        flash("Tên đăng nhập hoặc mật khẩu không đúng!", "error")
        return redirect(url_for('login'))
    
    return render_template('login.html')
##################

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Đăng ký tài khoản học sinh"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip()
        
        # Validation
        if not all([username, password, confirm_password, fullname]):
            flash("Vui lòng nhập đầy đủ thông tin!", "warning")
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash("Mật khẩu xác nhận không khớp!", "error")
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash("Mật khẩu phải có ít nhất 6 ký tự!", "warning")
            return redirect(url_for('register'))
        
        users_data = load_json("users.json")
        
        # Kiểm tra username đã tồn tại
        all_users = users_data.get('teachers', []) + users_data.get('students', [])
        if any(u['username'] == username for u in all_users):
            flash("Tên đăng nhập đã tồn tại!", "error")
            return redirect(url_for('register'))
        
        # Tạo tài khoản mới
        new_student = {
            "id": f"hs{len(users_data.get('students', [])) + 1:04d}",
            "username": username,
            "password": hash_password(password),
            "fullname": fullname,
            "email": email,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if 'students' not in users_data:
            users_data['students'] = []
        users_data['students'].append(new_student)
        
        if save_json("users.json", users_data):
            flash("Đăng ký thành công! Vui lòng đăng nhập.", "success")
            return redirect(url_for('login'))
        else:
            flash("Có lỗi xảy ra, vui lòng thử lại!", "error")
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Đăng xuất"""
    session.clear()
    flash("Đã đăng xuất thành công!", "success")
    return redirect(url_for('index'))

# ===============================================
# PHẦN 3: STUDENT ROUTES
# ===============================================

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    """Trang chủ học sinh - Hiển thị các khóa học"""
    if session.get('role') != 'student':
        return redirect(url_for('teacher_dashboard'))
    
    courses_data = load_json("courses.json")
    all_courses = courses_data.get('courses', [])
    
    # Lấy tiến độ học của học sinh
    student_id = session.get('user_id')
    progress_data = courses_data.get('student_progress', [])
    student_progress = [p for p in progress_data if p['student_id'] == student_id]
    
    # Tính toán tiến độ cho mỗi khóa học
    courses_with_progress = []
    for course in all_courses:
        course_lessons = [l for l in courses_data.get('lessons', []) 
                         if l['course_id'] == course['id']]
        total_lessons = len(course_lessons)
        
        completed_lessons = len([p for p in student_progress 
                                if p['course_id'] == course['id'] and p['completed']])
        
        progress_percent = (completed_lessons / total_lessons * 100) if total_lessons > 0 else 0
        
        courses_with_progress.append({
            **course,
            'total_lessons': total_lessons,
            'completed_lessons': completed_lessons,
            'progress_percent': round(progress_percent, 1)
        })
    
    return render_template('student_dashboard.html', 
                         courses=courses_with_progress,
                         fullname=session.get('fullname'))

@app.route('/student/course/<course_id>')
@login_required
def student_course_detail(course_id):
    """Chi tiết khóa học - Danh sách bài học"""
    if session.get('role') != 'student':
        flash("Bạn không có quyền truy cập!", "error")
        return redirect(url_for('index'))
    
    courses_data = load_json("courses.json")
    
    # Lấy thông tin khóa học
    course = next((c for c in courses_data.get('courses', []) 
                  if c['id'] == course_id), None)
    if not course:
        flash("Không tìm thấy khóa học!", "error")
        return redirect(url_for('student_dashboard'))
    
    # Lấy danh sách bài học
    lessons = [l for l in courses_data.get('lessons', []) 
              if l['course_id'] == course_id]
    lessons.sort(key=lambda x: x['order'])
    
    # Lấy tiến độ học
    student_id = session.get('user_id')
    progress_data = courses_data.get('student_progress', [])
    
    # Thêm thông tin completed cho mỗi bài học
    for lesson in lessons:
        progress = next((p for p in progress_data 
                        if p['student_id'] == student_id 
                        and p['lesson_id'] == lesson['id']), None)
        lesson['completed'] = progress['completed'] if progress else False
        lesson['score'] = progress.get('score', 0) if progress else 0
    
    return render_template('student_course_detail.html', 
                         course=course, 
                         lessons=lessons)

@app.route('/student/lesson/<lesson_id>')
@login_required
def student_lesson(lesson_id):
    """Xem bài học và làm bài tập"""
    if session.get('role') != 'student':
        flash("Bạn không có quyền truy cập!", "error")
        return redirect(url_for('index'))
    
    courses_data = load_json("courses.json")
    
    # Lấy thông tin bài học
    lesson = next((l for l in courses_data.get('lessons', []) 
                  if l['id'] == lesson_id), None)
    if not lesson:
        flash("Không tìm thấy bài học!", "error")
        return redirect(url_for('student_dashboard'))
    
    # Lấy thông tin khóa học
    course = next((c for c in courses_data.get('courses', []) 
                  if c['id'] == lesson['course_id']), None)
    
    # Lấy tiến độ
    student_id = session.get('user_id')
    progress_data = courses_data.get('student_progress', [])
    progress = next((p for p in progress_data 
                    if p['student_id'] == student_id 
                    and p['lesson_id'] == lesson_id), None)
    
    return render_template('student_lesson.html', 
                         lesson=lesson, 
                         course=course,
                         progress=progress)

@app.route('/student/lesson/<lesson_id>/quiz', methods=['GET', 'POST'])
@login_required
def student_quiz(lesson_id):
    """Làm bài tập trắc nghiệm"""
    if session.get('role') != 'student':
        flash("Bạn không có quyền truy cập!", "error")
        return redirect(url_for('index'))
    
    courses_data = load_json("courses.json")
    lesson = next((l for l in courses_data.get('lessons', []) 
                  if l['id'] == lesson_id), None)
    
    if not lesson:
        flash("Không tìm thấy bài học!", "error")
        return redirect(url_for('student_dashboard'))
    
    if request.method == 'POST':
        # Chấm điểm
        questions = lesson.get('quiz', [])
        correct = 0
        total = len(questions)
        
        for i, q in enumerate(questions):
            user_answer = request.form.get(f"q{i}", "").strip().upper()
            if user_answer == q['answer']:
                correct += 1
        
        score = (correct / total * 100) if total > 0 else 0
        
        # Lưu tiến độ
        student_id = session.get('user_id')
        progress_data = courses_data.get('student_progress', [])
        
        # Tìm hoặc tạo mới progress
        progress = next((p for p in progress_data 
                        if p['student_id'] == student_id 
                        and p['lesson_id'] == lesson_id), None)
        
        if progress:
            progress['score'] = score
            progress['completed'] = score >= 70  # Đạt nếu >= 70%
            progress['attempts'] = progress.get('attempts', 0) + 1
            progress['last_attempt'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            new_progress = {
                "student_id": student_id,
                "course_id": lesson['course_id'],
                "lesson_id": lesson_id,
                "score": score,
                "completed": score >= 70,
                "attempts": 1,
                "last_attempt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            progress_data.append(new_progress)
        
        courses_data['student_progress'] = progress_data
        save_json("courses.json", courses_data)
        
        flash(f"Bạn đạt {correct}/{total} câu đúng ({score:.1f}%)", "success")
        return redirect(url_for('student_lesson', lesson_id=lesson_id))
    
    # GET - Hiển thị bài tập
    questions = lesson.get('quiz', [])
    return render_template('student_quiz.html', lesson=lesson, questions=questions)

@app.route('/student/progress')
@login_required
def student_progress():
    """Xem tiến độ học tập của bản thân"""
    if session.get('role') != 'student':
        flash("Bạn không có quyền truy cập!", "error")
        return redirect(url_for('index'))
    
    courses_data = load_json("courses.json")
    student_id = session.get('user_id')
    
    # Lấy tất cả tiến độ của học sinh
    progress_data = [p for p in courses_data.get('student_progress', []) 
                    if p['student_id'] == student_id]
    
    # Nhóm theo khóa học
    courses_progress = {}
    for progress in progress_data:
        course_id = progress['course_id']
        if course_id not in courses_progress:
            course = next((c for c in courses_data.get('courses', []) 
                          if c['id'] == course_id), None)
            if course:
                courses_progress[course_id] = {
                    'course': course,
                    'lessons': []
                }
        
        # Lấy thông tin bài học
        lesson = next((l for l in courses_data.get('lessons', []) 
                      if l['id'] == progress['lesson_id']), None)
        if lesson:
            courses_progress[course_id]['lessons'].append({
                'lesson': lesson,
                'progress': progress
            })
    
    return render_template('student_progress.html', 
                         courses_progress=courses_progress)

# ===============================================
# PHẦN 4: TEACHER ROUTES
# ===============================================

@app.route('/teacher/dashboard')
@teacher_required
def teacher_dashboard():
    """Trang chủ giáo viên"""
    courses_data = load_json("courses.json")
    teacher_id = session.get('user_id')
    
    # Lấy các khóa học của giáo viên
    my_courses = [c for c in courses_data.get('courses', []) 
                  if c['teacher_id'] == teacher_id]
    
    # Thống kê
    total_courses = len(my_courses)
    total_lessons = sum(len([l for l in courses_data.get('lessons', []) 
                            if l['course_id'] == c['id']]) 
                       for c in my_courses)
    
    # Đếm số học sinh unique
    all_progress = courses_data.get('student_progress', [])
    course_ids = [c['id'] for c in my_courses]
    students_set = set(p['student_id'] for p in all_progress 
                      if p['course_id'] in course_ids)
    total_students = len(students_set)
    
    return render_template('teacher_dashboard.html',
                         courses=my_courses,
                         total_courses=total_courses,
                         total_lessons=total_lessons,
                         total_students=total_students)

@app.route('/teacher/course/create', methods=['GET', 'POST'])
@teacher_required
def teacher_create_course():
    """Tạo khóa học mới"""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        subject = request.form.get('subject', '').strip()
        description = request.form.get('description', '').strip()
        thumbnail = request.form.get('thumbnail', '').strip()
        
        if not title or not subject:
            flash("Vui lòng nhập đầy đủ thông tin!", "warning")
            return redirect(url_for('teacher_create_course'))
        
        courses_data = load_json("courses.json")
        
        new_course = {
            "id": f"course_{len(courses_data.get('courses', [])) + 1}",
            "teacher_id": session.get('user_id'),
            "teacher_name": session.get('fullname'),
            "title": title,
            "subject": subject,
            "description": description,
            "thumbnail": thumbnail,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if 'courses' not in courses_data:
            courses_data['courses'] = []
        courses_data['courses'].append(new_course)
        
        if save_json("courses.json", courses_data):
            flash("Tạo khóa học thành công!", "success")
            return redirect(url_for('teacher_course_detail', course_id=new_course['id']))
        else:
            flash("Có lỗi xảy ra!", "error")
    
    return render_template('teacher_create_course.html')

@app.route('/teacher/course/<course_id>')
@teacher_required
def teacher_course_detail(course_id):
    """Chi tiết khóa học - Quản lý bài học"""
    courses_data = load_json("courses.json")
    
    course = next((c for c in courses_data.get('courses', []) 
                  if c['id'] == course_id), None)
    
    if not course:
        flash("Không tìm thấy khóa học!", "error")
        return redirect(url_for('teacher_dashboard'))
    
    # Kiểm tra quyền sở hữu
    if course['teacher_id'] != session.get('user_id'):
        flash("Bạn không có quyền truy cập khóa học này!", "error")
        return redirect(url_for('teacher_dashboard'))
    
    # Lấy danh sách bài học
    lessons = [l for l in courses_data.get('lessons', []) 
              if l['course_id'] == course_id]
    lessons.sort(key=lambda x: x['order'])
    
    return render_template('teacher_course_detail.html', 
                         course=course, 
                         lessons=lessons)

####################
@app.route('/teacher/course/<course_id>/lesson/create', methods=['GET', 'POST'])
@teacher_required
def teacher_create_lesson(course_id):
    """Tạo bài học mới"""
    courses_data = load_json("courses.json")
    
    course = next((c for c in courses_data.get('courses', []) 
                  if c['id'] == course_id), None)
    
    if not course or course['teacher_id'] != session.get('user_id'):
        flash("Không có quyền!", "error")
        return redirect(url_for('teacher_dashboard'))
    
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        video_type = request.form.get('video_type', 'youtube')
        video_link = request.form.get('video_link', '').strip()
        document_link = request.form.get('document_link', '').strip()
        content = request.form.get('content', '').strip()
        
        if not title:
            flash("Vui lòng nhập tên bài học!", "warning")
            return redirect(url_for('teacher_create_lesson', course_id=course_id))
        
        # Xử lý link YouTube
        if video_type == 'youtube' and video_link:
            if 'youtu.be/' in video_link:
                video_id = video_link.split('youtu.be/')[-1].split('?')[0]
            elif 'watch?v=' in video_link:
                video_id = video_link.split('watch?v=')[-1].split('&')[0]
            else:
                video_id = video_link
            video_link = f"https://www.youtube.com/embed/{video_id}"
        
        # Xử lý link Google Drive - CHỈ KHI video_type == 'drive'
        if video_type == 'drive' and document_link:
            if '/file/d/' in document_link:
                file_id = document_link.split('/file/d/')[-1].split('/')[0]
                document_link = f"https://drive.google.com/file/d/{file_id}/preview"
        
        # Lấy câu hỏi trắc nghiệm
        quiz = []
        i = 0
        while True:
            question = request.form.get(f'question_{i}', '').strip()
            if not question:
                break
            
            # KIỂM TRA ĐẦY ĐỦ 4 ĐÁP ÁN
            option_a = request.form.get(f'option_A_{i}', '').strip()
            option_b = request.form.get(f'option_B_{i}', '').strip()
            option_c = request.form.get(f'option_C_{i}', '').strip()
            option_d = request.form.get(f'option_D_{i}', '').strip()
            answer = request.form.get(f'answer_{i}', '').strip().upper()
            
            # VALIDATION: Phải có đủ 4 đáp án và chọn đáp án đúng
            if not all([option_a, option_b, option_c, option_d, answer]):
                flash(f"Câu hỏi {i+1}: Vui lòng nhập đủ 4 đáp án và chọn đáp án đúng!", "warning")
                return redirect(url_for('teacher_create_lesson', course_id=course_id))
            
            if answer not in ['A', 'B', 'C', 'D']:
                flash(f"Câu hỏi {i+1}: Đáp án đúng phải là A, B, C hoặc D!", "warning")
                return redirect(url_for('teacher_create_lesson', course_id=course_id))
            
            quiz_item = {
                'question': question,
                'A': option_a,
                'B': option_b,
                'C': option_c,
                'D': option_d,
                'answer': answer
            }
            quiz.append(quiz_item)
            i += 1
        
        # KIỂM TRA: Phải có ít nhất 1 nội dung
        if not video_link and not document_link and not content and not quiz:
            flash("Vui lòng thêm ít nhất một nội dung: Video, Tài liệu, Nội dung hoặc Câu hỏi!", "warning")
            return redirect(url_for('teacher_create_lesson', course_id=course_id))
        
        # Tạo bài học mới
        lessons = [l for l in courses_data.get('lessons', []) 
                  if l['course_id'] == course_id]
        
        new_lesson = {
            "id": f"lesson_{len(courses_data.get('lessons', [])) + 1}",
            "course_id": course_id,
            "title": title,
            "order": len(lessons) + 1,
            "video_type": video_type,
            "video_link": video_link if video_type == 'youtube' else "",
            "document_link": document_link if video_type == 'drive' else "",
            "content": content,
            "quiz": quiz,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if 'lessons' not in courses_data:
            courses_data['lessons'] = []
        courses_data['lessons'].append(new_lesson)
        
        if save_json("courses.json", courses_data):
            flash("Tạo bài học thành công!", "success")
            return redirect(url_for('teacher_course_detail', course_id=course_id))
        else:
            flash("Có lỗi xảy ra!", "error")
    
    return render_template('teacher_create_lesson.html', course=course)
#################

@app.route('/teacher/lesson/<lesson_id>/edit', methods=['GET', 'POST'])
@teacher_required
def teacher_edit_lesson(lesson_id):
    """Chỉnh sửa bài học"""
    courses_data = load_json("courses.json")
    
    lesson = next((l for l in courses_data.get('lessons', []) 
                  if l['id'] == lesson_id), None)
    
    if not lesson:
        flash("Không tìm thấy bài học!", "error")
        return redirect(url_for('teacher_dashboard'))
    
    course = next((c for c in courses_data.get('courses', []) 
                  if c['id'] == lesson['course_id']), None)
    
    if not course or course['teacher_id'] != session.get('user_id'):
        flash("Không có quyền!", "error")
        return redirect(url_for('teacher_dashboard'))
    
    if request.method == 'POST':
      
        lesson['title'] = request.form.get('title', '').strip()
        lesson['video_type'] = request.form.get('video_type', 'youtube')
        lesson['video_link'] = request.form.get('video_link', '').strip()
        lesson['document_link'] = request.form.get('document_link', '').strip()
        lesson['content'] = request.form.get('content', '').strip()
        
        # Cập nhật quiz...
        
        if save_json("courses.json", courses_data):
            flash("Cập nhật bài học thành công!", "success")
            return redirect(url_for('teacher_course_detail', course_id=lesson['course_id']))
    
    return render_template('teacher_edit_lesson.html', lesson=lesson, course=course)

@app.route('/teacher/lesson/<lesson_id>/delete', methods=['POST'])
@teacher_required
def teacher_delete_lesson(lesson_id):
    """Xóa bài học"""
    courses_data = load_json("courses.json")
    
    lesson = next((l for l in courses_data.get('lessons', []) 
                  if l['id'] == lesson_id), None)
    
    if lesson:
        course = next((c for c in courses_data.get('courses', []) 
                      if c['id'] == lesson['course_id']), None)
        
        if course and course['teacher_id'] == session.get('user_id'):
            courses_data['lessons'] = [l for l in courses_data.get('lessons', []) 
                                      if l['id'] != lesson_id]
            save_json("courses.json", courses_data)
            flash("Đã xóa bài học!", "success")
            return redirect(url_for('teacher_course_detail', course_id=lesson['course_id']))
    
    flash("Không thể xóa bài học!", "error")
    return redirect(url_for('teacher_dashboard'))

@app.route('/teacher/course/<course_id>/students')
@teacher_required
def teacher_view_students(course_id):
    """Xem danh sách học sinh và tiến độ học"""
    courses_data = load_json("courses.json")
    users_data = load_json("users.json")
    
    course = next((c for c in courses_data.get('courses', []) 
                  if c['id'] == course_id), None)
    
    if not course or course['teacher_id'] != session.get('user_id'):
        flash("Không có quyền!", "error")
        return redirect(url_for('teacher_dashboard'))
    
    # Lấy tất cả bài học của khóa học
    lessons = [l for l in courses_data.get('lessons', []) 
              if l['course_id'] == course_id]
    total_lessons = len(lessons)
    
    # Lấy tiến độ học của các học sinh
    all_progress = courses_data.get('student_progress', [])
    course_progress = [p for p in all_progress if p['course_id'] == course_id]
    
    # Nhóm theo học sinh
    students_dict = {}
    for progress in course_progress:
        student_id = progress['student_id']
        if student_id not in students_dict:
            student = next((s for s in users_data.get('students', []) 
                          if s['id'] == student_id), None)
            if student:
                students_dict[student_id] = {
                    'info': student,
                    'completed': 0,
                    'total_score': 0,
                    'lessons_detail': []
                }
        
        if progress['completed']:
            students_dict[student_id]['completed'] += 1
        students_dict[student_id]['total_score'] += progress.get('score', 0)
        
        lesson = next((l for l in lessons if l['id'] == progress['lesson_id']), None)
        if lesson:
            students_dict[student_id]['lessons_detail'].append({
                'lesson': lesson,
                'progress': progress
            })
    
    # Tính điểm trung bình
    students_list = []
    for student_id, data in students_dict.items():
        avg_score = (data['total_score'] / total_lessons) if total_lessons > 0 else 0
        progress_percent = (data['completed'] / total_lessons * 100) if total_lessons > 0 else 0
        
        students_list.append({
            'info': data['info'],
            'completed': data['completed'],
            'total': total_lessons,
            'progress_percent': round(progress_percent, 1),
            'avg_score': round(avg_score, 1),
            'lessons_detail': data['lessons_detail']
        })
    
    # Sắp xếp theo tiến độ
    students_list.sort(key=lambda x: (-x['progress_percent'], -x['avg_score']))
    
    return render_template('teacher_view_students.html',
                         course=course,
                         students=students_list,
                         total_lessons=total_lessons)

@app.route('/teacher/student/<student_id>/detail')
@teacher_required
def teacher_student_detail(student_id):
    """Xem chi tiết tiến độ 1 học sinh qua các khóa học"""
    courses_data = load_json("courses.json")
    users_data = load_json("users.json")
    teacher_id = session.get('user_id')
    
    # Lấy thông tin học sinh
    student = next((s for s in users_data.get('students', []) 
                   if s['id'] == student_id), None)
    
    if not student:
        flash("Không tìm thấy học sinh!", "error")
        return redirect(url_for('teacher_dashboard'))
    
    # Lấy các khóa học của giáo viên
    my_courses = [c for c in courses_data.get('courses', []) 
                  if c['teacher_id'] == teacher_id]
    
    # Lấy tiến độ học sinh trong các khóa học của giáo viên
    student_progress = [p for p in courses_data.get('student_progress', []) 
                       if p['student_id'] == student_id 
                       and any(c['id'] == p['course_id'] for c in my_courses)]
    
    # Nhóm theo khóa học
    courses_detail = []
    for course in my_courses:
        lessons = [l for l in courses_data.get('lessons', []) 
                  if l['course_id'] == course['id']]
        course_progress = [p for p in student_progress 
                          if p['course_id'] == course['id']]
        
        if course_progress:  # Chỉ hiển thị khóa học mà học sinh đã tham gia
            total = len(lessons)
            completed = len([p for p in course_progress if p['completed']])
            avg_score = sum(p.get('score', 0) for p in course_progress) / total if total > 0 else 0
            
            courses_detail.append({
                'course': course,
                'total': total,
                'completed': completed,
                'progress_percent': round(completed / total * 100, 1) if total > 0 else 0,
                'avg_score': round(avg_score, 1),
                'lessons_progress': course_progress
            })
    
    return render_template('teacher_student_detail.html',
                         student=student,
                         courses=courses_detail)



@app.route('/api/courses')
def api_get_courses():
    """API lấy danh sách khóa học"""
    courses_data = load_json("courses.json")
    subject = request.args.get('subject', '')
    
    courses = courses_data.get('courses', [])
    
    if subject:
        courses = [c for c in courses if c['subject'] == subject]
    
    return jsonify({'success': True, 'courses': courses})

@app.route('/api/student/progress')
@login_required
def api_student_progress():
    """API lấy tiến độ học sinh"""
    if session.get('role') != 'student':
        return jsonify({'success': False, 'message': 'Unauthorized'}), 403
    
    courses_data = load_json("courses.json")
    student_id = session.get('user_id')
    
    progress = [p for p in courses_data.get('student_progress', []) 
               if p['student_id'] == student_id]
    
    return jsonify({'success': True, 'progress': progress})

@app.route('/api/teacher/statistics')
@teacher_required
def api_teacher_statistics():
    """API thống kê cho giáo viên"""
    courses_data = load_json("courses.json")
    teacher_id = session.get('user_id')
    
    my_courses = [c for c in courses_data.get('courses', []) 
                  if c['teacher_id'] == teacher_id]
    course_ids = [c['id'] for c in my_courses]
    
    all_progress = courses_data.get('student_progress', [])
    relevant_progress = [p for p in all_progress if p['course_id'] in course_ids]
    
    # Thống kê
    total_courses = len(my_courses)
    total_students = len(set(p['student_id'] for p in relevant_progress))
    total_lessons = sum(len([l for l in courses_data.get('lessons', []) 
                            if l['course_id'] == c['id']]) for c in my_courses)
    avg_completion = sum(1 for p in relevant_progress if p['completed']) / len(relevant_progress) * 100 if relevant_progress else 0
    
    return jsonify({
        'success': True,
        'statistics': {
            'total_courses': total_courses,
            'total_students': total_students,
            'total_lessons': total_lessons,
            'avg_completion': round(avg_completion, 1)
        }
    })
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

def initialize_data():
    """Khởi tạo tất cả file dữ liệu cần thiết"""
    init_users()
    init_courses()
    
    
    print("\n" + "="*70)
    print("THÔNG TIN TÀI KHOẢN GIÁO VIÊN MẶC ĐỊNH:")
    print("="*70)
    users_data = load_json("users.json")
    for teacher in users_data.get('teachers', []):
        print(f"Username: {teacher['username']} | Password: 123456 | Môn: {teacher['subject']}")
    print("="*70)
    print("Học sinh có thể đăng ký tài khoản mới tại /register")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Khởi tạo dữ liệu
    initialize_data()
    
    # Các phần còn lại giữ nguyên...
    for filename in ["data.json", "game.json"]:
        if not os.path.exists(filename):
            print(f"Tạo file {filename}...")
            save_json(filename, {"players": []})
    
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Đã tạo thư mục {PDF_FOLDER} để chứa file PDF")
    
    port = 5000
    
    print('=' * 70)
    print(f'Local URL: http://localhost:{port}')
    print(f'Network URL: http://0.0.0.0:{port}')
    print('=' * 70)
    print('Server đang chạy...')
    print('=' * 70)
    
    app.run(debug=True, host='0.0.0.0', port=port)

