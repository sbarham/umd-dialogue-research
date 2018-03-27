drop table if exists dialogue;
drop table if exists utterance;

create table dialogue (
  id integer primary key autoincrement,
  date real not null,
  network_type text,
  training_corpus text
);

create table utterance (
  id integer primary key autoincrement,
  dialogue_id integer,
  turn integer not null,
  ai boolean,
  text text,
  foreign key(dialogue_id) references dialogue(id)
);
