marked.setOptions({ breaks: true });

const form = document.getElementById("rag-form");
const output = document.getElementById("output");
const sourcesEl = document.getElementById("sources");

const searchStatus = document.getElementById("search-status");
const llmStatus = document.getElementById("llm-status");
const responseBlock = document.getElementById("response-block");
const sourcesBlock = document.getElementById("sources-block");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  output.innerHTML = "";
  sourcesEl.innerHTML = "";
  searchStatus.classList.remove("hidden", "status-done");
  llmStatus.classList.add("hidden");
  llmStatus.classList.remove("status-done");
  responseBlock.classList.add("hidden");
  sourcesBlock.classList.add("hidden");

  const query = new FormData(form).get("query");

  // Start progress feedback
  searchStatus.classList.remove("hidden");
  setTimeout(() => {
    searchStatus.classList.add("status-done");
    llmStatus.classList.remove("hidden");
  }, 200);

  const eventSource = new EventSource(`/stream?query=${encodeURIComponent(query)}`);
  let markdownBuffer = "";

  eventSource.onmessage = (event) => {
    if (event.data === "[DONE]") {
      eventSource.close();
      llmStatus.classList.add("status-done");
      responseBlock.classList.remove("hidden");
      sourcesBlock.classList.remove("hidden");

      fetch(`/sources?query=${encodeURIComponent(query)}`)
        .then(res => res.json())
        .then(urls => {
          urls.forEach(url => {
            const li = document.createElement("li");
            const link = document.createElement("a");
            link.href = url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = url;
            li.appendChild(link);
            sourcesEl.appendChild(li);
          });
        });
    } else {
      if (responseBlock.classList.contains("hidden")) {
        responseBlock.classList.remove("hidden");
      }

      if (event.data.length === 0) {
        markdownBuffer += "\n\n";
      } else {
        markdownBuffer += event.data;
      }
      output.innerHTML = marked.parse(markdownBuffer);
    }
  };
});

