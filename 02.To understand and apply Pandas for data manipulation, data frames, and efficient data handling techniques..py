import pandas as pd

print("\n" + "=" * 65)
print("AIM 2 : DATA MANIPULATION USING PANDAS")
print("=" * 65)

# Creating sample employee data
employee_data = {
    'Employee_Name': ['Rahul', 'Sneha', 'Amit', 'Neha', 'Karan'],
    'Age': [22, 29, 34, 41, 26],
    'Salary': [42000, 58000, 72000, 88000, 46000],
    'Department': ['Sales', 'IT', 'Finance', 'IT', 'Sales']
}

# Creating DataFrame
df = pd.DataFrame(employee_data)

print("\n📌 Employee DataFrame:")
print(df)

# Displaying DataFrame information
print("\n📌 DataFrame Information:")
df.info()

# Displaying descriptive statistics
print("\n📌 Descriptive Statistics:")
print(df.describe())

# Filtering employees from IT department
print("\n📌 Employees working in IT Department:")
print(df[df['Department'] == 'IT'])

# Filtering employees with salary greater than 60,000
print("\n📌 Employees with Salary greater than 60,000:")
print(df[df['Salary'] > 60000])

# Calculating average salary department-wise
print("\n📌 Average Salary by Department:")
print(df.groupby('Department')['Salary'].mean())

# Adding a new column for bonus (12% of salary)
df['Bonus'] = df['Salary'] * 0.12

print("\n📌 DataFrame after adding Bonus column:")
print(df)
