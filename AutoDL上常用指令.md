## 下载 claude code 

```bash
# 直接在虚拟环境安装
# 先装node，miniconda里直接装node
conda install nodejs -y
# 先确认node版本
node -v
# 安装claude-code
npm install -g @anthropic-ai/claude-code
```


## 配置 claude code 

- 全局：`~/.claude/settings.json`（推荐，不要提交到 git）
- 项目：`./.claude/settings.json`（项目本地）
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://vectide.cn",
    "ANTHROPIC_AUTH_TOKEN": "<API_KEY>",
    "ANTHROPIC_MODEL": "<MODEL_ID>",
    "ENABLE_TOOL_SEARCH": "true"
  }
}
```
懒人一键配置
```bash
mkdir -p .claude && cat > .claude/settings.json <<'EOF'
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN":  "<API_KEY>",
    "ANTHROPIC_BASE_URL": "https://vectide.cn",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash-0731",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro-0813",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro-0813",
    "ANTHROPIC_REASONING_MODEL": "deepseek-v4-pro-0813"
  },
  "models": {
    "deepseek-v4-flash-0731": {
      "max_context_tokens": 128000,
      "supports_tools": true,
      "supports_cache_control": false
    },
    "deepseek-v4-pro-0813": {
      "max_context_tokens": 128000,
      "supports_tools": true,
      "supports_cache_control": false
    }
  },
  "includeCoAuthoredBy": false
}
EOF
```

## 学术加速
```bash
source /etc/network_turbo
```
