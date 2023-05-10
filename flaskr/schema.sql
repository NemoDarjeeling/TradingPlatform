-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS admin;
DROP TABLE IF EXISTS trade;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  credit FLOAT NOT NULL
);
CREATE TABLE admin (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);
CREATE TABLE trade (
  tradeid INTEGER PRIMARY KEY AUTOINCREMENT,
  userid TEXT NOT NULL,
  shares INTEGER NOT NULL,
  types BINARY NOT NULL,
  ticker TEXT NOT NULL,
  dates DATE NOT NULL
);