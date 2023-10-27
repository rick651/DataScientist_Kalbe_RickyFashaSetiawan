-- Create Database
CREATE DATABASE Kalbe;

-- Import data
/* 
There are 4 dataset: 
  1. customer
  2. product
  3. store
  4. transaction
*/

-- Retrieve all records from the "customer" table
select *  from customer;

-- Update data type in Income Column from text to float
UPDATE customer SET Income = CAST(REPLACE(Income, ',', '.') AS Float);

-- Determine average age of customers based on their marital status
select `Marital Status`, 
	AVG(age) AS average_age
from customer
group by 1;

-- Determine average age of customers based on their gender
select case
    when gender = 0 then "Woman"
    when gender = 1 then "Man"
    else "-"
    end as Gender,
	avg(age) as average_age
from customer  
group by 1;

-- Determine the store name with the highest total quantity of sales
select transaction.StoreID, store.StoreName, store.GroupStore, 
	sum(transaction.Qty) as total_quantity 
from store join transaction 
on store.StoreID = transaction.StoreID
group by 1,2,3
order by total_quantity desc;

-- Identify the best-selling product by total amount of sales
select `Product Name`, 
	sum(totalamount) as  total_amount_Sales
from product
join transaction
on transaction.ProductID = product.ProductID
group by 1
order by 2 desc;
