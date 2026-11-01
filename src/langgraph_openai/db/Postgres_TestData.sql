INSERT INTO hr.employee (employee_id, name, dob) VALUES
    (1, 'Alice Johnson', '1990-01-15'),
    (2, 'Bob Smith', '1985-03-22'),
    (3, 'Carol Davis', '1992-07-09'),
    (4, 'David Brown', '1988-11-30'),
    (5, 'Emma Wilson', '1995-05-18'),
    (6, 'Frank Miller', '1983-09-02'),
    (7, 'Grace Lee', '1991-12-25'),
    (8, 'Henry Clark', '1987-02-14'),
    (9, 'Ivy Walker', '1993-06-07'),
    (10, 'Jack Hall', '1986-10-19'),
    (11, 'Karen Young', '1994-04-03'),
    (12, 'Liam King', '1989-08-27'),
    (13, 'Mia Wright', '1996-01-11'),
    (14, 'Noah Scott', '1984-03-29'),
    (15, 'Olivia Green', '1990-09-16'),
    (16, 'Paul Adams', '1982-12-05'),
    (17, 'Quinn Baker', '1997-06-21'),
    (18, 'Ruby Nelson', '1991-02-08'),
    (19, 'Sam Carter', '1988-07-24'),
    (20, 'Tina Mitchell', '1993-11-13')
ON CONFLICT (employee_id) DO NOTHING;

SELECT setval('hr.employee_employee_id_seq', (SELECT MAX(employee_id) FROM hr.employee));
