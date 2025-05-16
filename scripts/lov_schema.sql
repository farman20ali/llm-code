CREATE TABLE IF NOT EXISTS public.accident_types (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

INSERT INTO public.accident_types (label)
VALUES
    ('Minor Collision'),
    ('Major Collision'),
    ('Vehicle Rollover'),
    ('Hit and Run'),
    ('Pedestrian Accident')
ON CONFLICT (label) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.organizations (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    image_uri VARCHAR NULL,
    description TEXT NULL,
    phone VARCHAR NULL,
    location TEXT NULL,
    has_ambulance_service BOOLEAN DEFAULT TRUE,
    is_hospital BOOLEAN DEFAULT FALSE
);



CREATE TABLE IF NOT EXISTS user_organizations (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    organization_id INT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (organization_id) REFERENCES organizations(id)
);


-- Insert data into organizations table, ignoring duplicates
INSERT INTO public.organizations (label, image_uri, description, phone)
VALUES
    ('Rescue 1122', 'https://raw.githubusercontent.com/usamazahid/IRS/main/src/assets/img/rescue_1122.jpg', 'Emergency rescue service providing rapid response.', '+9201234567'),
    ('Edhi Foundation', 'https://raw.githubusercontent.com/usamazahid/IRS/main/src/assets/img/edhi_ambulance.jpg', 'A charitable organization offering emergency services.', '+9207654321'),
    ('Chhipa', 'https://raw.githubusercontent.com/usamazahid/IRS/main/src/assets/img/ambulance_icon.jpg', 'Provides ambulance and emergency services.', '+9201122334'),
    ('Suhayl Ambulance Service', 'https://raw.githubusercontent.com/usamazahid/IRS/main/src/assets/img/ambulance_icon.jpg', 'Offers medical emergency and ambulance services.', '+9205566778'),
    ('Gulab Devi Hospital Ambulance', 'https://raw.githubusercontent.com/usamazahid/IRS/main/src/assets/img/ambulance_icon.jpg', 'Hospital ambulance service providing critical care.', '+9209988776')
ON CONFLICT (label) DO NOTHING;

-- Create vehicle_involved table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.vehicle_involved (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description TEXT NULL
);

INSERT INTO public.vehicle_involved (label)
VALUES
    ('Pedestrian'),
    ('Bicycle'),
    ('Motorbike'),
    ('Truck'),
    ('Taxi'),
    ('Car'),
    ('Water Tanker'),
    ('Rickshaw/Chinqchi'),
    ('Dumper'),
    ('Trailer'),
    ('Loading Pickup'),
    ('Others')
ON CONFLICT (label) DO NOTHING;

-- Create patient_victim table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.patient_victim (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description TEXT NULL
);


INSERT INTO public.patient_victim (label)
VALUES
    ('Rider'),
    ('Pillion Rider'),
    ('Car/Taxi Driver'),
    ('Passenger'),
    ('Pedestrian'),
    ('Rickshaw/Chinqchi Driver'),
    ('Rickshaw/Chinqchi Passenger'),
    ('Others')
ON CONFLICT (label) DO NOTHING;


-- 

CREATE TABLE IF NOT EXISTS ambulance (
    ambulance_id SERIAL PRIMARY KEY,
    vehicle_number VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'available', --assigned, available, not-available,in progress
    organizations_id INT REFERENCES organizations(id)
);


CREATE TABLE IF NOT EXISTS ambulance_drivers (
    id SERIAL PRIMARY KEY,
    driver_id INT REFERENCES users(id) ON DELETE CASCADE,
    ambulance_id INT REFERENCES ambulance(ambulance_id) ON DELETE CASCADE,
    start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_date TIMESTAMP
);

-- Apparent Cause Table
CREATE TABLE apparent_cause (
    id SERIAL PRIMARY KEY,
    cause VARCHAR(50) NOT NULL,
    other_details VARCHAR(255)
);

INSERT INTO apparent_cause (cause) VALUES 
('Over speeding'),
('Brake Failure'),
('Road Condition'),
('Driver Negligence'),
('Weather Conditions'),
('Mechanical Failure'),
('Other');

-- Weather Condition Table
CREATE TABLE weather_condition (
    id SERIAL PRIMARY KEY,
    condition VARCHAR(50) NOT NULL
);

INSERT INTO weather_condition (condition) VALUES 
('Clear'),
('Rain'),
('Fog'),
('Dust'),
('Windy');

CREATE TABLE visibility (
    id SERIAL PRIMARY KEY,
    level VARCHAR(50) NOT NULL
);

INSERT INTO visibility (level) VALUES 
('Good'),
('Moderate'),
('Poor');

CREATE TABLE road_surface_condition (
    id SERIAL PRIMARY KEY,
    condition VARCHAR(50) NOT NULL
);

INSERT INTO road_surface_condition (condition) VALUES 
('Dry'),
('Wet'),
('Damaged'),
('Under Construction');

CREATE TABLE road_type (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL
);

INSERT INTO road_type (type) VALUES 
('Highway'),
('Urban Road'),
('Intersection'),
('Service Road'),
('Bridge/Flyover');


CREATE TABLE road_signage (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL
);

INSERT INTO road_signage (status) VALUES 
('Clear'),
('Faded'),
('Missing');

CREATE TABLE case_referred_to (
    id SERIAL PRIMARY KEY,
    unit VARCHAR(50) NOT NULL
);

INSERT INTO case_referred_to (unit) VALUES 
('Investigation Unit'),
('Traffic Police'),
('Legal Aid'),
('Not Applicable');

-- Preliminary Fault Assessment Table
CREATE TABLE preliminary_fault_assessment (
    id SERIAL PRIMARY KEY,
    fault VARCHAR(50) NOT NULL
);

INSERT INTO preliminary_fault_assessment (fault) VALUES 
('Driver 1'),
('Driver 2'),
('Road Condition'),
('Mechanical Failure'),
('Shared Fault'),
('Undetermined');


-- Create accident_types table if it doesn't exist
CREATE TABLE IF NOT EXISTS public.gender_types (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);



-- Insert data into gender_types table, ignoring duplicates
INSERT INTO public.gender_types (label)
VALUES
    ('female'),
    ('male'),
    ('other')
ON CONFLICT (label) DO NOTHING;



-- table vehicle condition
CREATE TABLE IF NOT EXISTS public.vehicle_condition (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);
 
INSERT INTO public.vehicle_condition (label)
VALUES
    ('Minor'),
    ('Major'),
    ('Total Loss')
ON CONFLICT (label) DO NOTHING;

-- table fitness_certificate_status
CREATE TABLE IF NOT EXISTS public.fitness_certificate_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

 
INSERT INTO public.fitness_certificate_status (label)
VALUES
    ('Valid'),
    ('Expired'),
    ('Not Available')
ON CONFLICT (label) DO NOTHING;


-- table causalities_status
CREATE TABLE IF NOT EXISTS public.causalities_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

 
INSERT INTO public.causalities_status (label)
VALUES
    ('Causalities'),
    ('Passengers'),
    ('Injured')
ON CONFLICT (label) DO NOTHING;


-- table injury_severity
CREATE TABLE IF NOT EXISTS public.injury_severity (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

 
INSERT INTO public.injury_severity (label)
VALUES
    ('Minor'),
    ('Major'),
    ('Fatal')
ON CONFLICT (label) DO NOTHING;


-- table road_tax_status
CREATE TABLE IF NOT EXISTS public.road_tax_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

 
INSERT INTO public.road_tax_status (label)
VALUES
    ('Paid'),
    ('Unpaid'),
    ('Pending')
ON CONFLICT (label) DO NOTHING;


-- table insurance_status
CREATE TABLE IF NOT EXISTS public.insurance_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

 
INSERT INTO public.insurance_status (label)
VALUES
    ('Active'),
    ('Expired'),
    ('Pending Renewal')
ON CONFLICT (label) DO NOTHING;