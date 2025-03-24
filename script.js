// script.js
function showAdminLogin() {
    document.getElementById('main-content').innerHTML = `
        <h2>🔐 Admin Login</h2>
        <input type="text" id="username" placeholder="Enter admin username" /><br>
        <input type="password" id="password" placeholder="Enter admin password" /><br>
        <button onclick="adminLogin()">Login</button>
        <div id="login-message"></div>
    `;
}

function showAttendance() {
    document.getElementById('main-content').innerHTML = `
        <h2>📷 Attendance Marking</h2>
        <button onclick="startCamera()">Start Camera</button>
        <video id="camera" width="320" height="240" autoplay></video>
        <div id="attendance-message"></div>
    `;
}

function adminLogin() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const message = document.getElementById('login-message');

    if (username === 'admin' && password === 'password') {
        message.innerText = '✅ Logged in as Admin!';
        showAdminOptions();
    } else {
        message.innerText = '❌ Invalid Credentials!';
    }
}

function showAdminOptions() {
    document.getElementById('main-content').innerHTML = `
        <h3>Admin Options</h3>
        <button onclick="addNewUser()">Add New User</button>
        <button onclick="viewAttendanceRecords()">View Attendance Records</button>
        <button onclick="generateReports()">Generate Reports</button>
    `;
}

function startCamera() {
    const video = document.getElementById('camera');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; })
        .catch(err => { console.error('Error accessing camera: ', err); });
}
