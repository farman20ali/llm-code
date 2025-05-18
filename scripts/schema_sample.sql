/* Traffic Accident Reporting System Schema and Sample Data */

-- SCHEMA DEFINITIONS (CREATE TABLE statements)

CREATE TABLE IF NOT EXISTS public.accident_types (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS public.vehicle_involved (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description TEXT NULL
);

CREATE TABLE IF NOT EXISTS public.patient_victim (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description TEXT NULL
);


CREATE TABLE apparent_cause (
    id SERIAL PRIMARY KEY,
    cause VARCHAR(50) NOT NULL,
    other_details VARCHAR(255)
);

CREATE TABLE weather_condition (
    id SERIAL PRIMARY KEY,
    condition VARCHAR(50) NOT NULL
);

CREATE TABLE visibility (
    id SERIAL PRIMARY KEY,
    level VARCHAR(50) NOT NULL
);

CREATE TABLE road_surface_condition (
    id SERIAL PRIMARY KEY,
    condition VARCHAR(50) NOT NULL
);

CREATE TABLE road_type (
    id SERIAL PRIMARY KEY,
    type VARCHAR(50) NOT NULL
);

CREATE TABLE road_signage (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL
);

CREATE TABLE case_referred_to (
    id SERIAL PRIMARY KEY,
    unit VARCHAR(50) NOT NULL
);

CREATE TABLE preliminary_fault_assessment (
    id SERIAL PRIMARY KEY,
    fault VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS public.gender_types (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS public.vehicle_condition (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS public.fitness_certificate_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS public.causalities_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS public.injury_severity (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS public.road_tax_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE IF NOT EXISTS public.insurance_status (
    id SERIAL PRIMARY KEY,
    label VARCHAR NOT NULL UNIQUE,
    description VARCHAR NULL
);

CREATE TABLE public.accident_reports (
	report_id serial4 NOT NULL,
	latitude numeric(9, 6) NULL,
	longitude numeric(9, 6) NULL,
	accident_location varchar(255) NULL,
    gis_coordinates GEOMETRY(Point, 4326) NULL,
	user_id int4 NULL,
	num_affecties int4 NULL,
	age int4 NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	status varchar(50) DEFAULT 'pending'::character varying NULL,
    severity int4 DEFAULT 1 NULL,
    image_uri text NULL,
	audio_uri text NULL,
    video_uri text NULL,
	description text NULL,
    officer_name VARCHAR(255),
    officer_designation VARCHAR(100),
    officer_contact_no VARCHAR(20),
    officer_notes TEXT,
    weather_condition  int4 NULL,
    visibility  int4 NULL,
    road_surface_condition  int4 NULL,
    road_type  int4 NULL,
    road_markings  int4 NULL,
    preliminary_fault  int4 NULL,
	gender  int4 NULL,
    cause  int4 NULL,
    vehicle_involved_id int4 NULL,
	patient_victim_id int4 NULL,
	accident_type_id int4 NULL,
	CONSTRAINT pk_accident_reports_report_id PRIMARY KEY (report_id),
	CONSTRAINT fk_accident_reports_accident_type_id FOREIGN KEY (accident_type_id) REFERENCES public.accident_types(id),
	CONSTRAINT fk_accident_reports_patient_victim_id FOREIGN KEY (patient_victim_id) REFERENCES public.patient_victim(id),
	CONSTRAINT fk_accident_reports_user_id FOREIGN KEY (user_id) REFERENCES public.users(id),
	CONSTRAINT fk_accident_reports_vehicle_involved_id FOREIGN KEY (vehicle_involved_id) REFERENCES public.vehicle_involved(id),
    CONSTRAINT fk_accident_reports_weather FOREIGN KEY (weather_condition) REFERENCES weather_condition(id),
    CONSTRAINT fk_accident_reports_visibility FOREIGN KEY (visibility) REFERENCES visibility(id),
    CONSTRAINT fk_accident_reports_road_surface FOREIGN KEY (road_surface_condition) REFERENCES road_surface_condition(id),
    CONSTRAINT fk_accident_reports_road_type FOREIGN KEY (road_type) REFERENCES road_type(id),
    CONSTRAINT fk_accident_reports_road_markings FOREIGN KEY (road_markings) REFERENCES road_signage(id),
    CONSTRAINT fk_accident_reports_preliminary_fault FOREIGN KEY (preliminary_fault) REFERENCES preliminary_fault_assessment(id),
    CONSTRAINT fk_accident_reports_gender_type FOREIGN KEY (gender) REFERENCES gender_types(id),
    CONSTRAINT fk_accident_reports_apparent_cause FOREIGN KEY (cause) REFERENCES apparent_cause(id)
);

CREATE TABLE public.accident_report_images (
    image_id SERIAL PRIMARY KEY,
    report_id INT NOT NULL,
    image_uri TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT accident_report_images_report_id_fkey FOREIGN KEY (report_id) REFERENCES public.accident_reports(report_id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS vehicle_details  (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES accident_reports(report_id) ON DELETE CASCADE,
    registration_no VARCHAR(50),
    type int4 NULL,
    condition VARCHAR(50),
    fitness_certificate_status VARCHAR(50),
    road_tax_status int4 NULL,
    insurance_status int4 NULL,
    CONSTRAINT fk_vehicle_details_vechile_type_id FOREIGN KEY (type) REFERENCES public.vehicle_involved(id) ON DELETE CASCADE,
    CONSTRAINT fk_vehicle_details_insurance_status FOREIGN KEY (insurance_status) REFERENCES public.insurance_status(id) ON DELETE CASCADE,
    CONSTRAINT fk_vehicle_details_road_tax_status FOREIGN KEY (road_tax_status) REFERENCES public.road_tax_status(id) ON DELETE CASCADE

);

CREATE TABLE IF NOT EXISTS driver_details (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES accident_reports(report_id) ON DELETE CASCADE,
    name VARCHAR(255),
    cnic_no VARCHAR(15),
    license_no VARCHAR(50),
    contact_no VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS passenger_casualties (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES accident_reports(report_id) ON DELETE CASCADE,
    type int4 NULL,
    name VARCHAR(255),
    hospital_name VARCHAR(255),
    injury_severity int4 NULL,
    CONSTRAINT fk_passenger_casualties_injury_severity FOREIGN KEY (injury_severity) REFERENCES injury_severity(id),
    CONSTRAINT fk_passenger_casualties_causalities_status FOREIGN KEY (type) REFERENCES causalities_status(id)
);

CREATE TABLE IF NOT EXISTS evidence_collection (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES accident_reports(report_id) ON DELETE CASCADE,
    photos BOOLEAN,
    videos BOOLEAN,
    sketch BOOLEAN
);

CREATE TABLE IF NOT EXISTS witness_details (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES accident_reports(report_id) ON DELETE CASCADE,
    name VARCHAR(255),
    contact_no VARCHAR(20),
    address TEXT
);

CREATE TABLE IF NOT EXISTS follow_up_actions (
    id SERIAL PRIMARY KEY,
    report_id INT REFERENCES accident_reports(report_id) ON DELETE CASCADE,
    fir_registered BOOLEAN,
    fir_number VARCHAR(50),
    challan_issued BOOLEAN,
    challan_number VARCHAR(50),
    case_referred_to int4 NULL,
    CONSTRAINT fk_follow_up_actions_case_referred FOREIGN KEY (case_referred_to) REFERENCES case_referred_to(id)
);

CREATE TABLE public.accident_vehicle_fitness (
    fitness_id SERIAL PRIMARY KEY,
    report_id INT NOT NULL,
    vehicle_no VARCHAR(50) NOT NULL,
    fitness_certificate_valid BOOLEAN NOT NULL,
    expiry_date DATE,
    road_tax_status int4 NULL,
    insurance_status int4 NULL,
    CONSTRAINT accident_vehicle_fitness_report_id_fkey FOREIGN KEY (report_id) REFERENCES public.accident_reports(report_id) ON DELETE CASCADE,
    CONSTRAINT fk_accident_vehicle_fitness_road_tax_status FOREIGN KEY (road_tax_status) REFERENCES road_tax_status(id),
    CONSTRAINT fk_accident_vehicle_fitness_insurance_status FOREIGN KEY (insurance_status) REFERENCES public.insurance_status(id) ON DELETE CASCADE

);


-- SAMPLE DATA (INSERT INTO statements)

INSERT INTO public.accident_types (label)
VALUES
    ('Minor Collision'),
    ('Major Collision'),
    ('Vehicle Rollover'),
    ('Hit and Run'),
    ('Pedestrian Accident')
ON CONFLICT (label) DO NOTHING;


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

INSERT INTO apparent_cause (cause) VALUES 
('Over speeding'),
('Brake Failure'),
('Road Condition'),
('Driver Negligence'),
('Weather Conditions'),
('Mechanical Failure'),
('Other');

INSERT INTO weather_condition (condition) VALUES 
('Clear'),
('Rain'),
('Fog'),
('Dust'),
('Windy');

INSERT INTO visibility (level) VALUES 
('Good'),
('Moderate'),
('Poor');

INSERT INTO road_surface_condition (condition) VALUES 
('Dry'),
('Wet'),
('Damaged'),
('Under Construction');

INSERT INTO road_type (type) VALUES 
('Highway'),
('Urban Road'),
('Intersection'),
('Service Road'),
('Bridge/Flyover');

INSERT INTO road_signage (status) VALUES 
('Clear'),
('Faded'),
('Missing');

INSERT INTO case_referred_to (unit) VALUES 
('Investigation Unit'),
('Traffic Police'),
('Legal Aid'),
('Not Applicable');

INSERT INTO preliminary_fault_assessment (fault) VALUES 
('Driver 1'),
('Driver 2'),
('Road Condition'),
('Mechanical Failure'),
('Shared Fault'),
('Undetermined');

INSERT INTO public.gender_types (label)
VALUES
    ('female'),
    ('male'),
    ('other')
ON CONFLICT (label) DO NOTHING;

INSERT INTO public.vehicle_condition (label)
VALUES
    ('Minor'),
    ('Major'),
    ('Total Loss')
ON CONFLICT (label) DO NOTHING;

INSERT INTO public.fitness_certificate_status (label)
VALUES
    ('Valid'),
    ('Expired'),
    ('Not Available')
ON CONFLICT (label) DO NOTHING;

INSERT INTO public.causalities_status (label)
VALUES
    ('Causalities'),
    ('Passengers'),
    ('Injured')
ON CONFLICT (label) DO NOTHING;

INSERT INTO public.injury_severity (label)
VALUES
    ('Minor'),
    ('Major'),
    ('Fatal')
ON CONFLICT (label) DO NOTHING;

INSERT INTO public.road_tax_status (label)
VALUES
    ('Paid'),
    ('Unpaid'),
    ('Pending')
ON CONFLICT (label) DO NOTHING;

INSERT INTO public.insurance_status (label)
VALUES
    ('Active'),
    ('Expired'),
    ('Pending Renewal')
ON CONFLICT (label) DO NOTHING;
