--************************************** Database and Deletions **************************************
-- Create database if it doesn't already exist
--CREATE DATABASE IF NOT EXISTS irs;

-- Drop tables if they exist to reset the structure
DROP TABLE IF EXISTS role_permissions CASCADE;
DROP TABLE IF EXISTS user_permissions CASCADE;
DROP TABLE IF EXISTS user_roles CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS roles CASCADE;
DROP TABLE IF EXISTS permissions CASCADE;

--************************************** Table Creation **************************************

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    mobile_number VARCHAR(255) NOT NULL UNIQUE,
    full_name VARCHAR(255),
    username VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    status VARCHAR(20) DEFAULT 'active', --inactive
    verification_status VARCHAR(20) DEFAULT 'N' ,   --N not verified and verified Y,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    device_token VARCHAR(255),
    device_id VARCHAR(255),
    designation VARCHAR(255) DEFAULT 'citizen'
);



-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    role_name VARCHAR(50) NOT NULL UNIQUE
);

-- Permissions table
CREATE TABLE IF NOT EXISTS permissions (
    id SERIAL PRIMARY KEY,
    permission_name VARCHAR(50) NOT NULL UNIQUE
);

-- User roles table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS user_roles (
    user_id INT,
    role_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (role_id) REFERENCES roles(id),
    PRIMARY KEY (user_id, role_id)
);

-- Role permissions table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS role_permissions (
    role_id INT,
    permission_id INT,
    FOREIGN KEY (role_id) REFERENCES roles(id),
    FOREIGN KEY (permission_id) REFERENCES permissions(id),
    PRIMARY KEY (role_id, permission_id)
);

-- User permissions table (many-to-many relationship)
CREATE TABLE IF NOT EXISTS user_permissions (
    user_id INT,
    permission_id INT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (permission_id) REFERENCES permissions(id),
    PRIMARY KEY (user_id, permission_id)
);


--************************************** Data Insertion **************************************

-- Insert Users
INSERT INTO users (username, password_hash, email, mobile_number,designation) VALUES 
    ('farman', 'test', 'admin@gmail.com', '03001','CEO'),
    ('test', 'test', 'citizen@gmail.com', '03002','citizen'),
    ('ambulance', 'test', 'ambulance@gmail.com', '03003','driver'),
    ('officer', 'test', 'officer@gmail.com', '03004','officer grade 2')
ON CONFLICT (username) DO NOTHING;

-- Insert Roles
INSERT INTO roles (role_name) VALUES 
    ('admin'), 
    ('citizen'), 
    ('ambulance'),
    ( 'officer')
ON CONFLICT (role_name) DO NOTHING;

-- Insert Permissions
INSERT INTO permissions (permission_name) VALUES 
    ('dashboard_view'), 
    ('profile_edit'), 
    ('report_accident'),
    ('view_profile'),
    ('call_ambulance'),
    ('view_history'),
    ('view_report'),
    ('manage_users'),
    ('pick_up'),
    ('drop_off'),
    ('alert_options'),
    ('manage_activities'),
    ('current_activities'),
    ('ambulance_stats'),
    ('search_users'),
    ('create_users'),
    ('update_users'),
    ('update_accident_report'),
    ('assign_ambulance'),
    ('authorized_login'),
    ('investigation_form'),
    ('view_offline_reports')
ON CONFLICT (permission_name) DO NOTHING;

-- Insert Role Permissions for "admin" role
INSERT INTO role_permissions (role_id, permission_id)
SELECT (SELECT id FROM roles WHERE role_name = 'admin'), id FROM permissions 
WHERE permission_name IN (
    'dashboard_view', 'authorized_login', 'ambulance_stats', 
    'view_profile', 'profile_edit', 'view_history', 'view_report', 
    'manage_activities', 'current_activities', 'assign_ambulance', 
    'update_accident_report', 'manage_users', 'search_users', 
    'create_users', 'update_users'
)
ON CONFLICT DO NOTHING;

-- Insert Role Permissions for "ambulance" role
INSERT INTO role_permissions (role_id, permission_id)
SELECT (SELECT id FROM roles WHERE role_name = 'ambulance'), id FROM permissions 
WHERE permission_name IN (
    'dashboard_view', 'authorized_login', 'view_profile', 
    'view_history', 'alert_options', 'drop_off', 
    'pick_up', 'update_accident_report'
)
ON CONFLICT DO NOTHING;

-- Insert Role Permissions for "citizen" role
INSERT INTO role_permissions (role_id, permission_id)
SELECT (SELECT id FROM roles WHERE role_name = 'citizen'), id FROM permissions 
WHERE permission_name IN (
    'dashboard_view', 'view_profile', 'view_history', 
    'profile_edit', 'call_ambulance', 'report_accident','view_offline_reports'
)
ON CONFLICT DO NOTHING;

-- Insert Role Permissions for "officer" role
INSERT INTO role_permissions (role_id, permission_id)
SELECT (SELECT id FROM roles WHERE role_name = 'officer'), id FROM permissions 
WHERE permission_name IN (
    'dashboard_view', 'view_profile', 'view_history', 
    'profile_edit', 'call_ambulance', 'investigation_form','view_offline_reports', 'view_report'
)
ON CONFLICT DO NOTHING;


-- Assign roles to users
INSERT INTO user_roles (user_id, role_id) VALUES 
    ((SELECT id FROM users WHERE username = 'farman'), (SELECT id FROM roles WHERE role_name = 'admin')),
    ((SELECT id FROM users WHERE username = 'test'), (SELECT id FROM roles WHERE role_name = 'citizen')),
    ((SELECT id FROM users WHERE username = 'ambulance'), (SELECT id FROM roles WHERE role_name = 'ambulance')),
    ((SELECT id FROM users WHERE username = 'officer'), (SELECT id FROM roles WHERE role_name = 'officer'))
ON CONFLICT DO NOTHING;

-- Assign direct permissions to users
INSERT INTO user_permissions (user_id, permission_id) VALUES 
    ((SELECT id FROM users WHERE username = 'test'), (SELECT id FROM permissions WHERE permission_name = 'report_accident'))
ON CONFLICT DO NOTHING;

-- Verify data insertion
--SELECT * FROM users;
--SELECT * FROM roles;
--SELECT * FROM permissions;
--SELECT * FROM role_permissions;
--SELECT * FROM user_roles;
--SELECT * FROM user_permissions;


--select queries