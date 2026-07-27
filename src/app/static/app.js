// ─────────────────────────────────────────────
// 카카오톡 스타일 채팅 페이지 로직
// ─────────────────────────────────────────────

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("sendBtn");

let threadId = null;

// ── 답변 생성 함수 ────────────────────────────
// 지금은 데모용 규칙 기반 답변입니다.
// 나중에 실제 모델 API(llama.cpp 서버 등)를 연결하려면
// 이 함수 내용만 fetch 호출로 바꾸면 됩니다. (아래 주석 예시 참고)
async function getBotReply(userText) {
  const res = await fetch("/agent/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: userText,
      thread_id: threadId,   // 첫 요청엔 null → 서버가 새로 발급
    }),
  });
  if (!res.ok) throw new Error(`서버 오류: ${res.status}`);
  const data = await res.json();
  threadId = data.thread_id;  // 세션 ID 저장 → 다음 요청부터 대화 이어짐
  return data.answer;
}

// ── 메시지 렌더링 ─────────────────────────────
function formatTime(date) {
  let h = date.getHours();
  const m = String(date.getMinutes()).padStart(2, "0");
  const period = h < 12 ? "오전" : "오후";
  h = h % 12 || 12;
  return `${period} ${h}:${m}`;
}

function addMessage(text, sender) {
  const row = document.createElement("div");
  row.className = `msg-row ${sender}`;

  const time = document.createElement("span");
  time.className = "msg-time";
  time.textContent = formatTime(new Date());

  if (sender === "bot") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🤖";

    const body = document.createElement("div");
    body.className = "msg-body";

    const name = document.createElement("span");
    name.className = "sender-name";
    name.textContent = "챗봇";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    body.append(name, bubble);
    row.append(avatar, body, time);
  } else {
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.append(bubble, time);
  }

  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

// "입력 중..." 말풍선 표시/제거
function showTyping() {
  const row = document.createElement("div");
  row.className = "msg-row bot typing";
  row.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="msg-body">
      <span class="sender-name">챗봇</span>
      <div class="bubble">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>
    </div>
  `;
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── 전송 흐름 ─────────────────────────────────
let isWaiting = false;

async function handleSend() {
  const text = inputEl.value.trim();
  if (!text || isWaiting) return;

  addMessage(text, "me");
  inputEl.value = "";
  autoResize();

  isWaiting = true;
  sendBtn.disabled = true;

  const typingRow = showTyping();

  try {
    const reply = await getBotReply(text);
    typingRow.remove();
    addMessage(reply, "bot");
  } catch (err) {
    typingRow.remove();
    addMessage("답변을 가져오지 못했어요. 잠시 후 다시 시도해 주세요.", "bot");
    console.error(err);
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// 입력창 높이 자동 조절 (여러 줄 입력 대응)
function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 96) + "px";
}

// ── 이벤트 바인딩 ─────────────────────────────
sendBtn.addEventListener("click", handleSend);

inputEl.addEventListener("keydown", (e) => {
  // Enter = 전송, Shift+Enter = 줄바꿈 (카톡 PC 버전과 동일)
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});

inputEl.addEventListener("input", autoResize);

// ── 초기 화면 ─────────────────────────────────
function init() {
  const divider = document.createElement("div");
  divider.className = "date-divider";
  const now = new Date();
  const days = ["일", "월", "화", "수", "목", "금", "토"];
  divider.textContent = `${now.getFullYear()}년 ${now.getMonth() + 1}월 ${now.getDate()}일 ${days[now.getDay()]}요일`;
  messagesEl.appendChild(divider);

  addMessage("안녕하세요! 메시지를 입력해 보세요 🙂", "bot");
  inputEl.focus();
}

init();
