import { AddCircle } from "@mui/icons-material";
import { Box, IconButton, TextField } from "@mui/material";
import { useState } from "react";
import { MuiList } from "./MuiList";
import { NavBar } from "./Navbar";

export const Home = () => {
    let [persons, setPersons] = useState([])
    let [newPerson, setNewPerson] = useState("");
    let addPerson = () => {
        setPersons([...persons,newPerson])
        setNewPerson('')
    }
    console.log(persons)
    console.log(newPerson)
  return (
    <>
      <NavBar />
      <MuiList name={persons}/>
      <Box
        sx={{
          display: "flex",
          width:"50vw",
          justifyContent: "space-between",
        //   padding:"4"
        }}
        pl={4}
        pr={4}
      >
        <TextField placeholder="Add a New Person" fullWidth value={newPerson} onChange={(event)=>{setNewPerson(event.target.value)}}/>
        <IconButton
          sx={{
            width: "50px",
            height: "50px",
            display: "flex",
            paddingLeft:"40px"
          }}
          onClick={()=>{addPerson()}}
        >
          <AddCircle
            color="primary"
            sx={{
              width: "50px",
              height: "50px",
              display: "flex",
            }}
          />
        </IconButton>
      </Box>
    </>
  );
};
