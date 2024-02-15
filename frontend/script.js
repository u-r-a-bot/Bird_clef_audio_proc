// app.js

const express = require('express');
const path = require('path');
const app = express();
const PORT = 3000;

// Set static folder for public assets
app.use(express.static(path.join(__dirname, 'public')));

// Define routes for each section
app.get('/home', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'home.html'));
});

app.get('/about', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'about.html'));
});

app.get('/references', (req, res) => {
    res.sendFile(path.join(__dirname, 'views', 'references.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
