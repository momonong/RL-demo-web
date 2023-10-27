# 使用官方Node.js基礎映像
FROM node:14

# 設定工作目錄
WORKDIR /app

# 首先只複製依賴描述文件
COPY package.json yarn.lock ./

# 安裝依賴
RUN yarn install

# 複製其他源碼和文件到工作目錄
COPY . .

# 設定你的app使用的端口
EXPOSE 3000

# 啟動React app的開發伺服器
CMD ["npm", "run", "dev"]

