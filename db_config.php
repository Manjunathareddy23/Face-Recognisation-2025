<?php
$host = 'localhost';
$user = 'root';
$password = '';
$dbname = 'face_attendance';

$conn = new mysqli($host, $user, $password, $dbname);
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$conn->query("CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    reg_number VARCHAR(50),
    image_path VARCHAR(255)
);");

$conn->query("CREATE TABLE IF NOT EXISTS attendance (
    reg_number VARCHAR(50),
    name VARCHAR(100),
    date DATE,
    time TIME
);");
?>
