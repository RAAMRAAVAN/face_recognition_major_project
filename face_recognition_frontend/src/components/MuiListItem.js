import { ExpandLess, ExpandMore, Mail, PermIdentityTwoTone } from "@mui/icons-material";
import {
  Avatar,
  Button,
  Collapse,
  Divider,
  List,
  ListItemAvatar,
  ListItemButton,
  ListItemIcon,
  ListItemText,
} from "@mui/material";
import { useState } from "react";
import { MuiImageList } from "./MuiImageList";

export const MuiListItem = (props) => {
  const [open, setOpen] = useState(false);
  const [imageListStatus, setImageListStatus] = useState(false);
  const handleClick = () => {
    setOpen(!open);
  };
  const setImageList = () => {
    setImageListStatus(!imageListStatus);
  };
  const personName = props.name
  return (
    <>
      <ListItemButton onClick={handleClick}>
        <ListItemIcon>
          <ListItemAvatar>
            <Avatar>
              <PermIdentityTwoTone />
            </Avatar>
          </ListItemAvatar>
        </ListItemIcon>
        <ListItemText primary={personName } />
        {open ? <ExpandLess /> : <ExpandMore />}
      </ListItemButton>
      <Collapse in={open} timeout="auto" unmountOnExit>
        <List component="div" disablePadding>
          <ListItemButton
            sx={{ pl: 4, justifyContent: "center" }}
            onClick={() => {
              setImageList();
            }}
          >
            View Uploaded Images
          </ListItemButton>
          {imageListStatus ? <MuiImageList /> : <></>}
          <ListItemButton sx={{ pl: 4, justifyContent: "center" }}>
            <Button variant="contained" component="label">
              Upload New Image
              <input hidden accept="image/*" multiple type="file" />
            </Button>
          </ListItemButton>
        </List>
      </Collapse>
      <Divider/>
    </>
  );
};
