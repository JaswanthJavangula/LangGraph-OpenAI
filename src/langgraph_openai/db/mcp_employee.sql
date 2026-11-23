-- Synthetic MCP test data only. Re-running preserves existing rows and values.
CREATE TABLE IF NOT EXISTS public.mcp_employee (
    employee_id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    dob DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO public.mcp_employee (employee_id, name, dob)
VALUES
    (1, 'Demo Employee One', DATE '1990-01-15'),
    (2, 'Demo Employee Two', DATE '1985-06-20'),
    (3, 'Demo Employee Three', DATE '1995-09-10'),
    (4, 'Demo Employee Four', DATE '1988-12-05'),
    (5, 'Demo Employee Five', NULL)
ON CONFLICT (employee_id) DO NOTHING;
