# WhatsApp Chat Analyzer

A modern, interactive web application that analyzes WhatsApp chat exports using machine learning algorithms. Built with Node.js, Python, and modern web technologies.

## 🚀 Features

- **Real-time Analysis**: Upload WhatsApp chat export files (.txt) and get instant analysis
- **Machine Learning**: Uses Random Forest classifier for user engagement pattern analysis
- **Interactive Dashboard**: Beautiful, responsive web interface with real-time results
- **User Insights**: Detailed analysis of message patterns, emoji usage, media sharing, and more
- **Modern UI/UX**: Gradient designs, smooth animations, and responsive layout

## 🛠️ Tech Stack

### Frontend
- **HTML5**: Semantic markup and modern structure
- **CSS3**: Advanced styling with gradients, animations, and responsive design
- **JavaScript (ES6+)**: Modern JavaScript with async/await and ES6+ features
- **Font Awesome**: Icon library for UI elements

### Backend
- **Node.js**: Server-side JavaScript runtime
- **Express.js**: Web application framework
- **Multer**: File upload handling middleware
- **CORS**: Cross-origin resource sharing support

### Machine Learning (Python)
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms (Random Forest)
- **emoji**: Emoji detection and analysis
- **regex**: Pattern matching for chat parsing

## 📁 Project Structure

```
wp_manoj/
├── public/                 # Frontend assets
│   ├── index.html         # Main HTML file
│   ├── styles.css         # CSS styles and animations
│   └── script.js          # Frontend JavaScript
├── server.js              # Node.js server
├── wp.py                  # Python ML analysis script
├── requirements.txt       # Python dependencies
├── package.json          # Node.js dependencies
└── README.md             # Project documentation
```

## 🚀 Installation

### Prerequisites
- **Node.js** (v14 or higher)
- **Python** (v3.8 or higher)
- **npm** or **yarn**

### Backend Setup

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **For production:**
   ```bash
   npm start
   ```

The server will run on `http://localhost:3000`

## 📱 Usage

### 1. Prepare Your Chat File
- Export your WhatsApp chat from the app
- Save as `.txt` file
- Ensure the file contains the standard WhatsApp export format

### 2. Upload and Analyze
- Open the web application in your browser
- Drag and drop your chat file or click to browse
- Wait for the analysis to complete
- View detailed results and insights

### 3. Understanding Results
- **Summary Statistics**: Total users, messages, and model accuracy
- **User Analysis**: Individual user engagement patterns
- **Message Patterns**: Average message length, emoji usage
- **Media Analysis**: Shared media and links count

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
PORT=3000
NODE_ENV=development
```

### Server Configuration
- **Port**: Default 3000 (configurable via environment)
- **File Upload**: Maximum file size limit
- **CORS**: Configured for development and production

## 📊 Analysis Features

### User Engagement Metrics
- Message count per user
- Average message length
- Emoji usage patterns
- Media sharing frequency
- Link sharing patterns

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Features**: Message patterns, timing, content analysis
- **Output**: User engagement classification (Active/Inactive)

## 🎨 UI Components

### Sections
- **Hero Section**: Gradient background with floating elements
- **Analyzer**: File upload and analysis interface
- **Results**: Interactive dashboard with charts and tables
- **About**: Technology stack and project information
- **Footer**: Contact information and social links

### Design Features
- **Gradient Color Schemes**: Modern, eye-catching designs
- **Smooth Animations**: Scroll-triggered and hover effects
- **Responsive Layout**: Mobile-first design approach
- **Glassmorphism**: Modern UI design elements

## 🔒 Security & Privacy

- **File Processing**: Chat files are processed locally and not stored permanently
- **Data Privacy**: No personal data is collected or stored
- **Secure Uploads**: File validation and size restrictions
- **HTTPS Ready**: Configured for secure connections

## 🚀 Deployment

### Local Development
```bash
npm run dev
```

### Production Build
```bash
npm start
```

### Docker (Optional)
```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Nandha M**
- LinkedIn: [@nandha-m-38681b250](https://www.linkedin.com/in/nandha-m-38681b250)
- GitHub: [@Nandha1218](https://github.com/Nandha1218)
- Email: nandhamarikannan2004@gmail.com
- Phone: +91 9789454161

## 🆘 Support

For support and questions:
- **Email**: nandhamarikannan2004@gmail.com
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check this README for common solutions

## 🔄 Changelog

### Version 1.0.0
- Initial release with core analysis functionality
- Modern, responsive web interface
- Machine learning-powered user engagement analysis
- Real-time file upload and processing
- Comprehensive results dashboard

## 📚 Additional Resources

- [WhatsApp Chat Export Guide](https://faq.whatsapp.com/android/chats/how-to-export-a-chat/)
- [Node.js Documentation](https://nodejs.org/docs/)
- [Express.js Guide](https://expressjs.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

**Made with ❤️ by Nandha M**
