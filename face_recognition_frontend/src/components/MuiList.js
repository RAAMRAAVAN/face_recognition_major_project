import React, { useState } from "react";
import {
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemAvatar,
  Avatar,
  ListItemButton,
  Divider,
  ListSubheader,
  Collapse,
  Button,
  SpeedDial,
  SpeedDialIcon,
  IconButton,
} from "@mui/material";
import {
  AddCircle,
  Drafts,
  ExpandLess,
  ExpandMore,
  Inbox,
  Mail,
  Send,
  StarBorder,
} from "@mui/icons-material";
import { MuiImageList } from "./MuiImageList";
import { MuiListItem } from "./MuiListItem";
export const MuiList = (props) => {
  const persons = props.name
  console.log("props",persons)
  return (
    <Box sx={{ width: "50vw", bgcolor: "#efefef" }}>
      <List
        sx={{ width: "100%", bgcolor: "background.paper" }}
        component="nav"
        aria-labelledby="nested-list-subheader"
        subheader={
          <ListSubheader component="div" id="nested-list-subheader">
            Added Peoples
          </ListSubheader>
        }
      >
        {persons.map((name, index)=>{
          return(<MuiListItem name={name} key={index}/>)
        })}
        
      </List>
    </Box>
  );
};
